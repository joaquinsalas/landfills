import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    # Se crea la clase. El 'smooth' es un número minúsculo para evitar que las matemáticas
    # exploten (como dividir entre cero) si la imagen está toda negra.
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    # Aquí es donde ocurre la magia cada vez que se calcula el error
    def forward(self, preds, targets):
        # preds: Lo que dibujó la IA.
        # targets: La máscara real perfecta.

        # 1. Sigmoid convierte los números raros de la IA en probabilidades del 0 al 1.
        preds = torch.sigmoid(preds)
        
        # 2. ".view(-1)" aplana las imágenes. Si era un cuadrado de 256x256, 
        # lo convierte en una tira larguísima de 65,536 píxeles formados uno tras otro.
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # 3. Calcula dónde se enciman ambos dibujos.
        # Al multiplicar, si la IA dijo 1 y la real es 1 (1x1=1). Si alguna falló, da 0.
        intersection = (preds * targets).sum()
        
        # 4. Aplica la fórmula matemática del Coeficiente de Dice.
        # (El doble de la intersección) dividido entre (la suma de todos los píxeles pintados).
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        # 5. Retorna "1 - dice".
        # Si el dibujo fue perfecto (dice = 1), el castigo es 0 (1 - 1 = 0). ¡Excelente!
        # Si el dibujo no se parece (dice = 0), el castigo es 1 (1 - 0 = 1). ¡Aprende más!
        return 1 - dice