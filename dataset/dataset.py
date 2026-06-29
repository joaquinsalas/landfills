import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class LandfillDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, img_norm=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.img_norm = img_norm

        # Lee nombres desde las máscaras para garantizar que existan los pares
        self.filenames = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(masks_dir)
            if f.endswith(('.png', '.tif'))
        ])
        print(f"  Dataset cargado: {len(self.filenames)} pares encontrados en {masks_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        # Carga tensor AlphaEarth [64, 512, 512]
        img_path = os.path.join(self.images_dir, f"{name}.pt")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No se encontró tensor .pt para: {name}")
        image = torch.load(img_path)

        # Busca máscara .png o .tif
        mask_path = None
        for ext in ['.png', '.tif']:
            candidate = os.path.join(self.masks_dir, f"{name}{ext}")
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            raise FileNotFoundError(f"No se encontró máscara para: {name}")

        # Carga y binariza máscara → [1, 512, 512]
        mask = Image.open(mask_path).convert("L")
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()
        mask = (mask > 127).float()

        # Fuerza tamaño exacto si es necesario
        if image.shape[-2:] != (512, 512):
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False
            ).squeeze(0)

        if mask.shape[-2:] != (512, 512):
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0), size=(512, 512), mode='nearest'
            ).squeeze(0)

        if self.transform:
            image, mask = self.transform(image, mask)
        if self.img_norm:
            image = self.img_norm(image)

        return image, mask


def get_dataloader(img_dir, mask_dir, batch_size, transform, img_norm, shuffle=True, num_workers=4):
    dataset = LandfillDataset(img_dir, mask_dir, transform=transform, img_norm=img_norm)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)