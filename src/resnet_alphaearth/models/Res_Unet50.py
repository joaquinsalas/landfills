import torch
import torch.nn as nn
# new
from torchvision.models import resnet50, ResNet50_Weights 
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Recibe la mitad de los canales + los canales del skip connection de ResNet
        self.conv1 = nn.Conv2d((in_channels // 2) + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Seguro contra pérdida de píxeles
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# new:
class ResUNet50(nn.Module): 
    def __init__(self):
        super().__init__()
        # 1. Descargamos el pre-entrenado
        # new
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT) 
        
        # 2. ENCODER Extraemos las capas clave de ResNet50
        self.encoder0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu) # Salida: 64 canales
        self.pool = base_model.maxpool 
        # new
        self.encoder1 = base_model.layer1 # Salida: 256 canales 
        self.encoder2 = base_model.layer2 # Salida: 512 canales 
        self.encoder3 = base_model.layer3 # Salida: 1024 canales 
        self.encoder4 = base_model.layer4 # Salida: 2048 canales (Fondo de la U)
        
        # 3. DECODER 
        # new : Modifiqué los in_channels y skip_channels de cada bloque para soportar la profundidad de la ResNet50
        self.dec4 = DecoderBlock(in_channels=2048, skip_channels=1024, out_channels=512)
        self.dec3 = DecoderBlock(in_channels=512, skip_channels=512, out_channels=256)
        self.dec2 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        self.dec1 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        
        # 4. CAPA FINAL 
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) 
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        # Bajada (Encoder)
        e0 = self.encoder0(x) 
        e1 = self.encoder1(self.pool(e0)) 
        e2 = self.encoder2(e1) 
        e3 = self.encoder3(e2) 
        e4 = self.encoder4(e3) 
        
        # Subida (Decoder) (skip connections)
        d4 = self.dec4(e4, e3) 
        d3 = self.dec3(d4, e2) 
        d2 = self.dec2(d3, e1) 
        d1 = self.dec1(d2, e0) 
        
        out = self.final_up(d1) 
        out = self.final_conv(out)
        return out