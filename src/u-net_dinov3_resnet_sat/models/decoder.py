# models/decoder.py
# ─────────────────────────────────────────────────────────────────────
# Decoder con bloques residuales (ResBlock + UpBlock) que reincorpora
# skip connections REALES de ResNet50 (layer1 y layer2) durante el
# upsampling, en lugar de subir resolucion "a ciegas" como en la
# version solo-DINO.
#
# Flujo:
#   fused        (B, 512, 14x14)
#     -> UpBlock + skip layer2 (512ch @ 28x28) -> (B, 256, 28x28)
#     -> UpBlock + skip layer1 (256ch @ 56x56) -> (B, 128, 56x56)
#     -> UpBlock (sin skip)                    -> (B,  64, 112x112)
#     -> UpBlock (sin skip)                    -> (B,  32, 224x224)
#     -> head                                  -> (B,   1, 224x224)
# ─────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Bloque residual con dos convoluciones y batch normalization."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Proyeccion para ajustar canales si cambian (shortcut connection)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) \
                    if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)                # Shortcut
        out = F.relu(self.bn1(self.conv1(x)))  # Conv1 + BN + ReLU
        out = self.bn2(self.conv2(out))        # Conv2 + BN
        return F.relu(out + residual)          # Suma residual + ReLU


class UpBlock(nn.Module):
    """
    Upsample x2 + (opcional) concatenar skip connection de ResNet + ResBlock.

    in_ch:   canales del feature map que se esta subiendo
    skip_ch: canales del skip de ResNet a concatenar (0 si no hay skip)
    out_ch:  canales de salida del bloque
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block = ResBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # fusion por concatenacion
        return self.block(x)


class DualDecoder(nn.Module):
    """
    Decoder completo del modelo dual-encoder.
    Entrada: feature map fusionado (B, 512, 14, 14) + skips de ResNet
    Salida:  mascara (B, 1, 224, 224)
    """
    def __init__(self):
        super().__init__()
        self.up1 = UpBlock(512, 512, 256)  # 14->28  + skip ResNet layer2 (512ch)
        self.up2 = UpBlock(256, 256, 128)  # 28->56  + skip ResNet layer1 (256ch)
        self.up3 = UpBlock(128,   0,  64)  # 56->112 sin skip
        self.up4 = UpBlock(64,    0,  32)  # 112->224 sin skip

        # Cabeza final: genera mascara binaria de 1 canal
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16,  1, 1),
            nn.Sigmoid()   # Salida entre 0 y 1 (probabilidad de ser relleno)
        )

    def forward(self, fused, skip_layer1, skip_layer2):
        x = self.up1(fused, skip_layer2)  # 14 -> 28  con skip ResNet layer2
        x = self.up2(x,     skip_layer1)  # 28 -> 56  con skip ResNet layer1
        x = self.up3(x,     None)         # 56 -> 112
        x = self.up4(x,     None)         # 112 -> 224
        return self.head(x)
