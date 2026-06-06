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
        if tokens.dim() == 3:
            B, L, C = tokens.shape
            if L == h * w + 1:
                tokens = tokens[:, 1:, :]
            elif L != h * w:
                tokens = tokens[:, :h*w, :]
            tokens = tokens.reshape(B, h, w, C).permute(0, 3, 1, 2)
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
        
        self.vit.load_state_dict(state, strict=False)
        print(f"Pesos de DINOv3 ViT-7B cargados exitosamente desde {ckpt_path}")

        # Activation Checkpointing obligatorio para contener el mapa de activaciones gigante
        if hasattr(self.vit, 'set_grad_checkpointing'):
            self.vit.set_grad_checkpointing(True)
            
        self.embed_dim = 4096

        # Forzar pesos del encoder a precisiones eficientes de 16-bits
        if torch.cuda.is_bf16_supported():
            self.vit.to(torch.bfloat16)
            self.dtype = torch.bfloat16
        else:
            self.vit.to(torch.float16)
            self.dtype = torch.float16

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        x_enc = x.to(self.dtype)
        features = self.vit(x_enc)
        
        # Casting a float32 para la estabilidad numérica del decoder U-Net
        features = [f.float() for f in features]

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
        embed_dim = self.encoder.embed_dim
        dec_ch = decoder_channels

        self.tok2map = nn.ModuleList([
            TokenToMap(embed_dim, dec_ch[0]),
            TokenToMap(embed_dim, dec_ch[1]),
            TokenToMap(embed_dim, dec_ch[2]),
            TokenToMap(embed_dim, dec_ch[3]),
        ])

        self.dec0 = DecoderBlock(dec_ch[0], dec_ch[1], dec_ch[1])
        self.dec1 = DecoderBlock(dec_ch[1], dec_ch[2], dec_ch[2])
        self.dec2 = DecoderBlock(dec_ch[2], dec_ch[3], dec_ch[3])
        self.dec3 = DecoderBlock(dec_ch[3], 0,         dec_ch[4])
        self.dec4 = DecoderBlock(dec_ch[4], 0,         dec_ch[4])

        self.head = nn.Conv2d(dec_ch[4], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        features, h_p, w_p = self.encoder(x)

        f0 = self.tok2map[3](features[0], h_p, w_p)
        f1 = self.tok2map[2](features[1], h_p, w_p)
        f2 = self.tok2map[1](features[2], h_p, w_p)
        f3 = self.tok2map[0](features[3], h_p, w_p)

        d0 = self.dec0(f3, f2)
        d1 = self.dec1(d0, f1)
        d2 = self.dec2(d1, f0)
        d3 = self.dec3(d2)
        
        out = self.head(d3)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out
