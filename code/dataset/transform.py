import torchvision.transforms.v2 as transforms
import torch

def get_transforms(input_size):

    # define las transformaciones que se le haran a las imagenes y mascaras
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True), 
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    # normalizacion solo se aplica a la IMAGEN, no a la mascara
    img_normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    return train_transform, val_test_transform, img_normalization
