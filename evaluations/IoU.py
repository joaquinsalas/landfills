import torch

def calculate_iou(preds, masks, threshold=0.5):
    # threshold=0.5 significa que si la IA está un 50% segura o más de que hay basura,
    # lo pintará de blanco. Si está menos segura, lo dejará negro.
    
    # Convierte las predicciones de la IA en blanco y negro puros (True o False)
    preds = torch.sigmoid(preds) > threshold
    masks = masks > threshold # Asegura que la máscara real también sea blanco/negro puro
    
    # Intersección: Cuenta cuántos píxeles blancos coinciden exactamente en ambos dibujos.
    intersection = (preds & masks).float().sum((1, 2, 3))
    
    # Unión: Cuenta cuántos píxeles blancos hay en total si juntas los dos dibujos.
    union = (preds | masks).float().sum((1, 2, 3))
    
    # Calcula el porcentaje de éxito (Intersección dividida entre Unión)
    # Se le suma un 1e-6 (un número enanito) por si la imagen no tiene nada de basura, 
    # para que la computadora no intente dividir cero entre cero y marque error.
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Te entrega el promedio del IoU de las fotos que procesó
    return iou.mean().item()