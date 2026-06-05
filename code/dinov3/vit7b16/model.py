"""
DINOv3 ViT-B/16 → U-Net Decoder
Segmentación semántica binaria sobre imágenes satelitales/SAR.

Arquitectura:
  Encoder : DINOv3 ViT-B/16 (patch=16, embed_dim=768)
             → extrae features de 4 capas intermedias del transformer
  Neck    : Reshape de tokens a mapas 2D + proyección de canal
  Decoder : 4 etapas de upsampling (×2 cada una) con skip connections
  Head    : Conv 1×1 → 1 canal (logit binario)

Para imágenes >1024px se usa tiling en inferencia (ver infer.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Proyección de tokens ViT → mapa 2D
# ---------------------------------------------------------------------------

class TokenToMap(nn.Module):
    """
    Toma los patch tokens (B, N, C) y los reformea a (B, C_out, H/s, W/s).
    s = patch_size del ViT (16).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # tokens: (B, H_p*W_p, C)  donde H_p = H/patch, W_p = W/patch
        x = self.proj(tokens)               # (B, N, out_dim)
        B, N, C = x.shape
        x = x.permute(0, 2, 1)             # (B, C, N)
        x = x.reshape(B, C, h, w)          # (B, C, H_p, W_p)
        return x


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Upsample ×2, concatena skip, doble conv.
    Si no hay skip, solo sube y procesa.
    """
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


# ---------------------------------------------------------------------------
# Encoder wrapper: DINOv3 ViT-B/16
# ---------------------------------------------------------------------------

class DinoEncoder(nn.Module):
    """
    Carga el checkpoint y expone los features de 4 bloques del transformer.

    Bloques elegidos para ViT-B (12 bloques en total):
        índices  2,  5,  8, 11  (cada 3 bloques → 4 escalas)

    Los tokens del CLS se descartan; solo se usan los patch tokens.
    """

    INTERMEDIATE_LAYERS = [9, 19, 29, 39]

    def __init__(self, ckpt_path: str, freeze: bool = True):
        super().__init__()
        # Cambiar el identificador del modelo base por el de DINOv3 7B satelital
        self.vit = timm.create_model(
            "vit_7b_patch16_dinov3.sat493m",
            pretrained=False,
            num_classes=0,
            global_pool="",
        )
        
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            if "model" in state:
                state = state["model"]
            self.vit.load_state_dict(state, strict=False)
            print(f"Pesos de DINOv3 ViT-7B cargados exitosamente desde {ckpt_path}")

        self.freeze = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        x : (B, C, H, W)  — debe ser divisible por patch_size=16
        Devuelve lista de 4 tensores de tokens (B, N, 768).
        """
        B, _, H, W = x.shape
        patch_size = 16
        h_p = H // patch_size
        w_p = W // patch_size

        # Patch embedding
        tokens = self.vit.patch_embed(x)                # (B, N, 768)

        # Añadir CLS token + pos embedding
        cls = self.vit.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Interpolar pos_embed si la resolución difiere del preentrenamiento
        pos_embed = self.vit.pos_embed                  # (1, 1+196, 768)
        if tokens.shape[1] != pos_embed.shape[1]:
            pos_embed = self._interpolate_pos_embed(pos_embed, h_p, w_p)
        tokens = tokens + pos_embed

        tokens = self.vit.pos_drop(tokens)

        features = []
        for i, block in enumerate(self.vit.blocks):
            tokens = block(tokens)
            if i in self.INTERMEDIATE_LAYERS:
                # Excluir CLS token → solo patch tokens
                features.append(tokens[:, 1:, :])

        return features, h_p, w_p   # lista de 4 × (B, h_p*w_p, 768)

    @staticmethod
    def _interpolate_pos_embed(pos_embed, h_p, w_p):
        """Interpolación bicúbica del positional embedding."""
        cls_pos  = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]
        orig_size = int(patch_pos.shape[1] ** 0.5)
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h_p, w_p), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h_p * w_p, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)


# ---------------------------------------------------------------------------
# U-Net completa
# ---------------------------------------------------------------------------

class DinoUNet(nn.Module):
    """
    DINOv3 ViT-B/16 como encoder + decoder U-Net de 4 etapas.

    Canales del decoder (por defecto):
        f3 (prof 11) → 256
        f2 (prof  8) → 128
        f1 (prof  5) →  64
        f0 (prof  2) →  32
        + una etapa final ×2 sin skip →  16

    La salida es un logit (sin sigmoid); usa BCEWithLogitsLoss.
    """

    def __init__(
        self,
        ckpt_path: str,
        num_classes: int = 1,
        freeze_encoder: bool = True,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
    ):
        super().__init__()

        self.encoder = DinoEncoder(ckpt_path, freeze=freeze_encoder)
        embed_dim = self.encoder.embed_dim  # 768

        dec_ch = decoder_channels           # (256, 128, 64, 32, 16)

        # Proyecciones token→mapa para cada escala
        self.tok2map = nn.ModuleList([
            TokenToMap(embed_dim, dec_ch[0]),   # f3 → profundidad máxima
            TokenToMap(embed_dim, dec_ch[1]),   # f2
            TokenToMap(embed_dim, dec_ch[2]),   # f1
            TokenToMap(embed_dim, dec_ch[3]),   # f0
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

        self.tok2map = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(4096),
                nn.Linear(4096, 64)
            ),
            nn.Sequential(
                nn.LayerNorm(4096),
                nn.Linear(4096, 128)
            ),
            nn.Sequential(
                nn.LayerNorm(4096),
                nn.Linear(4096, 128)
            ),
            nn.Sequential(
                nn.LayerNorm(4096),
                nn.Linear(4096, 256)
            ),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, _, H, W = x.shape

        # ── Encoder ──────────────────────────────────────────────────────
        features, h_p, w_p = self.encoder(x)
        # features[0..3] = capas [2,5,8,11], cada una (B, h_p*w_p, 768)

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


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"

    model = DinoUNet(ckpt_path=ckpt, num_classes=1, freeze_encoder=True)
    model.eval()

    # Simula un tile SAR 512×512 de 3 canales
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")   # debe ser (1, 1, 512, 512)
    print("✓ Forward pass OK")
