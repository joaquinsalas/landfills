import torch
import torch.nn as nn
from DoubleConv import DoubleConv
from encoder import Down
from decoder import Up

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # Define cuántos canales entran (3 canales para fotos RGB) y cuántas clases salen (1 clase: relleno sanitario)
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Primer filtro apenas entra la imagen
        self.inc = DoubleConv(n_channels, 64)
        
        # El Encoder (la bajada): La imagen se hace más pequeña pero los canales (filtros) aumentan
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024) # Punto más profundo de la red
        
        # El Decoder (la subida): La imagen vuelve a crecer recuperando los detalles
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Capa final: Aplana los 64 filtros restantes en un solo mapa de salida
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    # La ruta exacta que recorre la información
    def forward(self, x):
        # Baja guardando copias en cada paso (x1, x2, x3...)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Sube utilizando la información actual y las copias anteriores (Skip Connections)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Te entrega la predicción final
        logits = self.outc(x)
        return logits