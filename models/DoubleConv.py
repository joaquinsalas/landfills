import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # nn.Sequential empaqueta varios pasos en uno solo
        self.double_conv = nn.Sequential(
            # Paso 1: Primer filtro (Conv2d)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # Paso 2: Normalización (BatchNorm2d) para que la red no se confunda con colores muy brillantes
            nn.BatchNorm2d(out_channels),
            # Paso 3: Función de activación (ReLU) que descarta la información inútil (valores negativos)
            nn.ReLU(inplace=True),
            
            # Paso 4, 5 y 6: Repite el proceso exacto una vez más
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # Función que empuja los datos a través de los pasos anteriores
    def forward(self, x):
        return self.double_conv(x)