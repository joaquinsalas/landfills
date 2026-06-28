# models/backbone.py
# ─────────────────────────────────────────────────────────────────────
# Dos encoders complementarios:
#   DinoEncoder   -> DINOv3-vitl16 (o DINOv2-large como fallback)
#                    features globales/semanticas, mapa 14x14 (1024 ch)
#   ResNetEncoder -> ResNet50 preentrenado en ImageNet
#                    features espaciales/de borde en 4 escalas:
#                    layer1: 56x56 (256ch)  layer2: 28x28 (512ch)
#                    layer3: 14x14 (1024ch) layer4:  7x7  (2048ch)
# ─────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models

from config.config import MODELO_ENCODER, MODELO_FALLBACK, TOKEN_HF


# ══════════════════════════════════════════════════════════════════════
#  ENCODER 1: DINOv3 / DINOv2  (contexto global y semantica)
# ══════════════════════════════════════════════════════════════════════
class DinoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Intentar cargar DINOv3, si falla usar DINOv2-large
        try:
            self.backbone = AutoModel.from_pretrained(
                MODELO_ENCODER,
                token=TOKEN_HF,
                output_hidden_states=True
            )
            self.dim  = 1024  # DINOv3-vitl16 usa 1024 dimensiones
            self.name = "DINOv3-vitl16"
            print(f"DinoEncoder: {self.name} cargado")
        except Exception as e:
            print(f"  DINOv3 no disponible ({e})\n  Usando fallback: DINOv2-large")
            self.backbone = AutoModel.from_pretrained(
                MODELO_FALLBACK,
                output_hidden_states=True
            )
            self.dim  = 1024
            self.name = "DINOv2-large (fallback)"
            print(f"DinoEncoder: {self.name} cargado")

        # Congelar encoder al inicio del entrenamiento
        self.frozen = True
        for param in self.backbone.parameters():
            param.requires_grad = False

    def descongelar(self, ultimas_capas=4):
        """Descongela las ultimas N capas del transformer para fine-tuning."""
        self.frozen = False
        try:
            capas = self.backbone.encoder.layer
            for capa in capas[-ultimas_capas:]:
                for param in capa.parameters():
                    param.requires_grad = True
            print(f"DinoEncoder: {ultimas_capas} capas descongeladas")
        except AttributeError:
            # Si la estructura no es estandar, descongelar todo
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("DinoEncoder: completamente descongelado")

    def forward(self, x):
        """Retorna la lista de hidden_states de todas las capas del transformer."""
        if self.frozen:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        else:
            outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        return outputs.hidden_states  # lista de tensores (B, tokens, dim)


# ══════════════════════════════════════════════════════════════════════
#  ENCODER 2: ResNet50  (detalle espacial y bordes)
# ══════════════════════════════════════════════════════════════════════
class ResNetEncoder(nn.Module):
    """
    ResNet50 preentrenado en ImageNet usado como encoder convolucional.
    Expone 4 niveles de skip connections, utiles para reconstruir
    detalle espacial fino que DINOv3 (basado en patches de 16px) no captura:

        layer1 -> 256  canales @ 56x56
        layer2 -> 512  canales @ 28x28
        layer3 -> 1024 canales @ 14x14   (se fusiona con DINOv3)
        layer4 -> 2048 canales @  7x7
    """
    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base    = models.resnet50(weights=weights)

        # Extraer capas individuales para obtener los feature maps intermedios
        self.stem   = nn.Sequential(base.conv1, base.bn1,
                                     base.relu, base.maxpool)  # 224 -> 56
        self.layer1 = base.layer1   # 56x56 -> 256 canales
        self.layer2 = base.layer2   # 28x28 -> 512 canales
        self.layer3 = base.layer3   # 14x14 -> 1024 canales
        self.layer4 = base.layer4   #  7x7  -> 2048 canales

        self.out_channels = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048,
        }
        print("ResNetEncoder: ResNet50 preentrenado (ImageNet) cargado")

    def forward(self, x):
        """Retorna diccionario con los feature maps de cada capa."""
        x  = self.stem(x)     # (B,  64, 56, 56)
        s1 = self.layer1(x)   # (B, 256, 56, 56)
        s2 = self.layer2(s1)  # (B, 512, 28, 28)
        s3 = self.layer3(s2)  # (B,1024, 14, 14)
        s4 = self.layer4(s3)  # (B,2048,  7,  7)

        return {'layer1': s1, 'layer2': s2,
                'layer3': s3, 'layer4': s4}
