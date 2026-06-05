import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class TokenToMap(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # Si timm devuelve una secuencia plana de tokens (B, L, C)
        if tokens.dim() == 3:
            B, L, C = tokens.shape
            if L == h * w + 1:
                tokens = tokens[:, 1:, :]  # Remover token CLS si existe
            elif L != h * w:
                tokens = tokens[:, :h*w, :]
            tokens = tokens.reshape(B, h, w, C).permute(0, 3, 1, 2)
            
        # Si timm ya devuelve un mapa de características 2D (B, C, H, W)
        elif tokens.dim() == 4:
            if tokens.shape[1] != self.proj.in_channels and tokens.shape[3] == self.proj.in_channels:
                tokens = tokens.permute(0, 3, 1, 2)
                
        return self.proj(tokens)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # ajuste por si las dimensiones difieren 1px
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DinoEncoder(nn.Module):

    INTERMEDIATE_LAYERS = [9, 19, 29, 39]

    def __init__(self, ckpt_path: str, freeze: bool = True):
        super().__init__()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(state, dict):
            for key in ("model", "state_dict", "teacher", "student"):
                if key in state:
                    state = state[key]
                    break

        import timm
        self.vit = timm.create_model(
            "vit_7b_patch16_dinov3.sat493m",
            pretrained=False,
            features_only=True,           
            out_indices=(9, 19, 29, 39), 
        )
        
        missing, unexpected = self.vit.load_state_dict(state, strict=False)
        print(f"Pesos de DINOv3 ViT-7B cargados exitosamente desde {ckpt_path}")
        self.embed_dim = 4096

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        features = self.vit(x)
        if features[0].dim() == 4:
            h_p, w_p = features[0].shape[-2:]
        else:
            h_p, w_p = x.shape[-2] // 16, x.shape[-1] // 16
        
        return features, h_p, w_p


class DinoUNet(nn.Module):

    def __init__(
        self,
        ckpt_path: str,
        num_classes: int = 1,
        freeze_encoder: bool = True,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
    ):
        super().__init__()

        self.encoder = DinoEncoder(ckpt_path, freeze=freeze_encoder)
        embed_dim = self.encoder.embed_dim  # 4096

        dec_ch = decoder_channels           # (256, 128, 64, 32, 16)

        # Proyecciones token→mapa mapeando correctamente a los canales de la U-Net
        self.tok2map = nn.ModuleList([
            TokenToMap(embed_dim, dec_ch[0]),   # f3 → profundidad máxima (256 canales)
            TokenToMap(embed_dim, dec_ch[1]),   # f2 → (128 canales)
            TokenToMap(embed_dim, dec_ch[2]),   # f1 → (64 canales)
            TokenToMap(embed_dim, dec_ch[3]),   # f0 → (32 canales)
        ])

        # Decoder (de profundo a superficial)
        # Etapa 0: bottleneck → ×2, luego cat con f2_map
        self.dec0 = DecoderBlock(dec_ch[0], dec_ch[1], dec_ch[1])
        # Etapa 1: ×2, cat con f1_map
        self.dec1 = DecoderBlock(dec_ch[1], dec_ch[2], dec_ch[2])
        # Etapa 2: ×2, cat con f0_map
        self.dec2 = DecoderBlock(dec_ch[2], dec_ch[3], dec_ch[3])
        # Etapa 3: ×2, sin skip (ya no hay features del encoder)
        self.dec3 = DecoderBlock(dec_ch[3], 0,         dec_ch[4])
        # Etapa 4: ×2 final (patch_size=16 → necesitamos ×16 total: 2^4=16 ✓)
        self.dec4 = DecoderBlock(dec_ch[4], 0,         dec_ch[4])

        # Cabeza de segmentación
        self.head = nn.Conv2d(dec_ch[4], num_classes, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, _, H, W = x.shape

        # ── Encoder ──────────────────────────────────────────────────────
        features, h_p, w_p = self.encoder(x)

        # ── Neck: token → mapa 2D ────────────────────────────────────────
        f0 = self.tok2map[3](features[0], h_p, w_p)  # capa 2  (B, 32, h_p, w_p)
        f1 = self.tok2map[2](features[1], h_p, w_p)  # capa 5  (B, 64, h_p, w_p)
        f2 = self.tok2map[1](features[2], h_p, w_p)  # capa 8  (B, 128, h_p, w_p)
        f3 = self.tok2map[0](features[3], h_p, w_p)  # capa 11 (B, 256, h_p, w_p)

        # ── Decoder ──────────────────────────────────────────────────────
        d0 = self.dec0(f3, f2)   # (B, 128, h_p*2, w_p*2)
        d1 = self.dec1(d0, f1)   # (B,  64, h_p*4, w_p*4)
        d2 = self.dec2(d1, f0)   # (B,  32, h_p*8, w_p*8)
        d3 = self.dec3(d2)       # (B,  16, h_p*16, w_p*16)  → resolución original
        d4 = self.dec4(d3)       # extra ×2 si quieres super-resolución (opcional)

        # Ajuste fino a tamaño original (por si H/W no es múltiplo exacto)
        out = self.head(d3)      # usa d3 si quieres salida en resolución original
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"

    model = DinoUNet(ckpt_path=ckpt, num_classes=1, freeze_encoder=True)
    model.eval()

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")   # debe ser (1, 1, 512, 512)
    print("✓ Forward pass OK")
