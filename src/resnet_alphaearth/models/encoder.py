import torch
import torch.nn as nn
from DoubleConv import DoubleConv

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # MaxPool2d recorta la imagen a la mitad de su tamaño reteniendo solo los píxeles más importantes
            nn.MaxPool2d(2),
            # Llama al bloque de la sección 4 para analizar esta nueva imagen encogida
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)