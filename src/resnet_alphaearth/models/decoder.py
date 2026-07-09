#decoder orginal
import torch
import torch.nn as nn
import torch.nn.functional as F # <-- Agregado para poder redimensionar si es necesario

# Este bloque toma la salida profunda de la U-Net
# la hace más grande 
# la junta con la skip connection del encoder
# y luego la refina con convoluciones
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        
        # ConvTranspose2d:
        # duplica alto y ancho del mapa de características.
        # Ejemplo: de 32x32 pasa a 64x64.
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )
        
        # Primera convolucion despues de concatenar
        # recibe el doble de canales porque aquí se juntan
        # 1. lo que viene del decoder
        # 2. la skip connection del encoder
        self.conv1 = nn.Conv2d(
            out_channels * 2,
            out_channels,
            kernel_size=3,
            padding=1
        )
        
        # Normalizacion 1: 
        # Estabiliza el aprendizaje para que no se confunda con valores muy altos.
        # Esta es la pieza exacta que le faltaba a tu archivo .pth
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Activacion no lineal
        self.relu1 = nn.ReLU(inplace=True)
        
        # Segunda convolucion para seguir refinando
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
        
        # Normalizacion 2:
        # Mantiene los datos controlados antes de pasar a la siguiente capa
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Segunda activacion
        self.relu2 = nn.ReLU(inplace=True)

    # Funcion de seguridad:
    # A veces, al dividir la imagen en el encoder, se pierde 1 pixel por el redondeo.
    # Esta funcion revisa si miden exactamente lo mismo, y si no, estira un poco 
    # la skip_connection para que encajen a la perfeccion y no marque error.
    def resize_skip_connection(self, skip_connection, target_size):
        if skip_connection.shape[-2:] == target_size:
            return skip_connection
        return F.interpolate(skip_connection, size=target_size, mode="bilinear", align_corners=False)

    def forward(self, x, skip_connection):
        # 1. sube la resolucion
        x = self.upsample(x)

        # 1.5 asegura que los tamaños sean identicos antes de juntarlos
        skip_connection = self.resize_skip_connection(skip_connection, x.shape[-2:])

        # 2. concatena con la skip connection
        # dim=1 = dimensión de canales
        x = torch.cat([x, skip_connection], dim=1)

        # 3. refinamiento con convoluciones
        x = self.conv1(x)
        x = self.bn1(x) # <-- aplica la primera normalizacion
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x) # <-- aplica la segunda normalizacion
        x = self.relu2(x)

        # 4. devuelve la salida del bloque decoder
        return x