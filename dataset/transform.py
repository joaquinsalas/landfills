import torchvision.transforms.v2 as transforms
import torch

def get_transforms(input_size):

    # transformaciones
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)), # Redimensiona las imágenes a un tamaño uniforme para que la IA pueda procesarlas
        transforms.RandomHorizontalFlip(p=0.5),# Voltea la imagen horizontalmente 
        transforms.RandomVerticalFlip(p=0.5),# Voltea la imagen verticalmente
        #new
        transforms.RandomRotation(degrees=45), # Rota la imagen en ángulos aleatorios
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Simula diferentes horas del sol
        #----
        transforms.ToImage(), # Convierte el tensor de vuelta a imagen para aplicar las siguientes transformaciones
        transforms.ToDtype(torch.float32, scale=True), # Convierte a float32 y normaliza los valores de 0-255 a 0-1 para que la IA los entienda mejor
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    # normaliza 
    img_normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    return train_transform, val_test_transform, img_normalization
