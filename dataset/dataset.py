import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms

class LandfillDataset(Dataset):
    # Constructor: Se ejecuta cuando creas al "bibliotecario" y se le dice dónde buscar
    def __init__(self, images_dir, masks_dir, transform=None, img_norm=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform # Transformaciones matemáticas (como rotar fotos)
        self.img_norm = img_norm # Normalización (ajustar colores)
        
        # Lee los nombres de todos los archivos en la carpeta y los ordena
        self.filenames = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.tif'))])

    # Le dice al sistema cuántas fotos tienes en total
    def __len__(self):
        return len(self.filenames)

    # El metodo mas importante: extrae exactamente la foto y la máscara que le pidas
    def __getitem__(self, idx):
        # Arma la ruta completa del archivo
        img_path = os.path.join(self.images_dir, self.filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.filenames[idx])

        # Abre la foto a color (RGB) y la máscara en blanco y negro (L)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Si le programaste transformaciones o normalizaciones, se las aplica aquí
        if self.transform:
            image, mask = self.transform(image, mask)
        if self.img_norm:
            image = self.img_norm(image)
            
        return image, mask # Te entrega el paquete listo para entrenar

# Esta funcion crea un "cargador". Agrupa las fotos de 2 en 2 (tu BATCH_SIZE) y las revuelve si se lo pides
def get_dataloader(img_dir, mask_dir, batch_size, transform, img_norm, shuffle=True, num_workers=4):
    dataset = LandfillDataset(img_dir, mask_dir, transform=transform, img_norm=img_norm)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)