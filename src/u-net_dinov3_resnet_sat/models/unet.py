# models/unet.py
# ─────────────────────────────────────────────────────────────────────
# DualEncoderUNet: fusiona DINOv3 (contexto global/semantica) con
# ResNet50 (detalle espacial/bordes) y usa skip connections REALES
# de ResNet en el decoder.
#
# Arquitectura:
#
#   Imagen (224x224)
#     ├── DinoEncoder   -> hidden_states -> mapa espacial (B,1024,14,14)
#     └── ResNetEncoder -> layer1 (B, 256, 56,56)
#                          layer2 (B, 512, 28,28)
#                          layer3 (B,1024, 14,14)  <- se fusiona con DINOv3
#                          layer4 (B,2048,  7, 7)  (no usado en esta version)
#
#   Fusion @ 14x14:
#     proj_dino(1024->256) + proj_resnet(1024->256) -> concat(512) -> fusion
#
#   Decoder con skip connections reales:
#     fused(14x14) -> up1 + skip_layer2(28x28)
#                  -> up2 + skip_layer1(56x56)
#                  -> up3(112x112) -> up4(224x224) -> mascara
# ─────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import DinoEncoder, ResNetEncoder
from models.decoder  import DualDecoder


class DualEncoderUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ── Encoder 1: DINOv3 (o DINOv2 como fallback) ────────────────
        self.dino = DinoEncoder()
        self.dim  = self.dino.dim  # 1024

        # ── Encoder 2: ResNet50 preentrenado en ImageNet ──────────────
        self.resnet = ResNetEncoder(pretrained=True)

        # ── Proyecciones para fusionar DINOv3 + ResNet en 14x14 ───────
        # DINOv3 (ultima capa): 1024 ch @ 14x14
        # ResNet layer3:        1024 ch @ 14x14
        self.proj_dino   = nn.Conv2d(self.dim, 256, 1)  # 1024 -> 256
        self.proj_resnet = nn.Conv2d(1024,     256, 1)  # 1024 -> 256

        # ── Proyecciones de las skip connections de ResNet ────────────
        self.proj_skip2 = nn.Conv2d(512, 512, 1)  # layer2: 512 ch @ 28x28
        self.proj_skip1 = nn.Conv2d(256, 256, 1)  # layer1: 256 ch @ 56x56

        # ── Fusionador: concat(dino 256 + resnet 256) -> 512 ──────────
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # ── Decoder con skip connections reales ───────────────────────
        self.decoder = DualDecoder()

    def descongelar_encoder(self, ultimas_capas=4):
        """Descongela las ultimas N capas de DINOv3 para fine-tuning.
        El ResNet50 se mantiene siempre entrenable (es mucho mas
        liviano que el transformer y aporta el detalle espacial fino)."""
        self.dino.descongelar(ultimas_capas)

    def _dino_patch_map(self, hidden_states, layer_idx):
        """
        Convierte los patch tokens de DINOv3/DINOv2 a un mapa espacial
        (B, dim, H, W), ignorando el token CLS y los register tokens.

        DINOv3: 1 CLS + 4 Register + 196 Patch = 201 tokens -> grid 14x14
        DINOv2: 1 CLS + 256 Patch              = 257 tokens -> grid 16x16
        """
        B          = hidden_states[layer_idx].shape[0]
        num_tokens = hidden_states[layer_idx].shape[1]

        if num_tokens == 201:    # DINOv3-vitl16
            n_patches, grid = 196, 14
        elif num_tokens == 257:  # DINOv2-large
            n_patches, grid = 256, 16
        else:                    # Fallback generico
            n_patches = int(round((num_tokens - 1) ** 0.5)) ** 2
            grid      = int(n_patches ** 0.5)

        tokens = hidden_states[layer_idx][:, -n_patches:, :]  # (B, n_patches, dim)
        return tokens.permute(0, 2, 1).reshape(B, self.dim, grid, grid)

    def forward(self, x):
        # ── Encoder DINOv3 ─────────────────────────────────────────────
        hidden_states = self.dino(x)
        num_layers    = len(hidden_states) - 1
        dino_map      = self._dino_patch_map(hidden_states, num_layers)  # (B,1024,14,14)

        # ── Encoder ResNet50 ───────────────────────────────────────────
        resnet_feats = self.resnet(x)
        rn_layer1    = resnet_feats['layer1']  # (B, 256, 56, 56)
        rn_layer2    = resnet_feats['layer2']  # (B, 512, 28, 28)
        rn_layer3    = resnet_feats['layer3']  # (B,1024, 14, 14)

        # Si el grid de DINOv2 (16x16) no coincide con ResNet layer3 (14x14),
        # se reescala el mapa de DINO al tamaño del feature map de ResNet.
        if dino_map.shape[-2:] != rn_layer3.shape[-2:]:
            dino_map = F.interpolate(
                dino_map, size=rn_layer3.shape[-2:],
                mode='bilinear', align_corners=False)

        # ── Fusion @ 14x14: DINOv3 (semantica) + ResNet (espacial) ─────
        dino_proj   = self.proj_dino(dino_map)     # (B, 256, 14, 14)
        resnet_proj = self.proj_resnet(rn_layer3)  # (B, 256, 14, 14)
        fused = self.fusion(
            torch.cat([dino_proj, resnet_proj], dim=1))  # (B, 512, 14, 14)

        # ── Skip connections de ResNet, proyectadas ────────────────────
        skip2 = self.proj_skip2(rn_layer2)  # (B, 512, 28, 28)
        skip1 = self.proj_skip1(rn_layer1)  # (B, 256, 56, 56)

        # ── Decoder con skip connections reales ────────────────────────
        mask = self.decoder(fused, skip1, skip2)

        # Asegurar tamaño exacto de salida 224x224
        return F.interpolate(mask, size=(224, 224), mode='bilinear', align_corners=False)


# Alias para mantener compatibilidad con train.py / test.py / plots.py
UNetDINO = DualEncoderUNet


if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DualEncoderUNet().to(device)
    dummy  = torch.randn(2, 3, 224, 224).to(device)
    out    = model(dummy)
    print(f"\nTest forward -> input: {dummy.shape} -> output: {out.shape}")
    assert out.shape == (2, 1, 224, 224), "Error: forma de salida inesperada"
    print("Test OK")
