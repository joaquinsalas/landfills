import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Sube la resolución a la mitad de los canales
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

class ResUNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Descargamos el cerebro maestro pre-entrenado
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 2. ENCODER Extraemos las capas clave de ResNet18
        self.encoder0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu) # Salida: 64 canales
        self.pool = base_model.maxpool 
        self.encoder1 = base_model.layer1 # Salida: 64 canales
        self.encoder2 = base_model.layer2 # Salida: 128 canales
        self.encoder3 = base_model.layer3 # Salida: 256 canales
        self.encoder4 = base_model.layer4 # Salida: 512 canales (Fondo de la U)
        
        # 3. DECODER (Construido a la medida de ResNet)
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        
        # 4. CAPA FINAL (Sube la última vez para igualar el 256x256 original)
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) 
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        # Bajada (Encoder)
        e0 = self.encoder0(x) 
        e1 = self.encoder1(self.pool(e0)) 
        e2 = self.encoder2(e1) 
        e3 = self.encoder3(e2) 
        e4 = self.encoder4(e3) 
        
        # Subida (Decoder) conectando los cables (skip connections)
        d4 = self.dec4(e4, e3) 
        d3 = self.dec3(d4, e2) 
        d2 = self.dec2(d3, e1) 
        d1 = self.dec1(d2, e0) 
        
        out = self.final_up(d1) 
        out = self.final_conv(out)
        return out