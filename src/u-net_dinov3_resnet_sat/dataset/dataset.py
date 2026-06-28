# dataset/dataset.py
# ─────────────────────────────────────────────────────────────────────
# Clase LandfillDataset: carga imagenes y mascaras desde disco,
# aplica resize, augmentation y normalizacion.
# ─────────────────────────────────────────────────────────────────────
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config.config import IMG_SIZE, NORM_MEAN, NORM_STD, EXTENSIONES


class LandfillDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=IMG_SIZE, augment=True):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.size     = size
        self.augment  = augment  # Si aplica data augmentation o no

        # Lista de imagenes validas en la carpeta
        self.imagenes = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(EXTENSIONES)
        ])

        if not self.imagenes:
            raise FileNotFoundError(f"No se encontraron imagenes en: {img_dir}")

        self.resize    = T.Resize((size, size))   # Redimensiona a IMG_SIZE x IMG_SIZE
        self.normalize = T.Normalize(mean=NORM_MEAN, std=NORM_STD)  # Normaliza con valores ImageNet

    def __len__(self):
        return len(self.imagenes)  # Total de imagenes en el dataset

    def _buscar_mascara(self, nombre):
        # Busca la mascara con cualquier extension valida
        for ext in EXTENSIONES:
            mp = os.path.join(self.mask_dir, nombre + ext)
            if os.path.exists(mp):
                return Image.open(mp).convert("L")  # L = escala de grises
        raise FileNotFoundError(f"Mascara no encontrada para: {nombre}")

    def _augmentar(self, image, mask):
        # Volteo horizontal al azar con 50% de probabilidad
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Volteo vertical al azar con 50% de probabilidad
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # Rotacion aleatoria de 0, 90, 180 o 270 grados
        angle = random.choice([0, 90, 180, 270])
        image = TF.rotate(image, angle)
        mask  = TF.rotate(mask, angle)

        # Cambios de brillo y contraste solo en la imagen (no en la mascara)
        image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))

        return image, mask

    def __getitem__(self, idx):
        nombre   = os.path.splitext(self.imagenes[idx])[0]  # Nombre sin extension
        img_path = os.path.join(self.img_dir, self.imagenes[idx])

        # Cargar y convertir imagen a RGB
        image = Image.open(img_path).convert("RGB")
        mask  = self._buscar_mascara(nombre)

        # Redimensionar imagen y mascara al mismo tamaño
        image = self.resize(image)
        mask  = self.resize(mask)

        # Aplicar augmentation si esta activado
        if self.augment and random.random() > 0.5:
            image, mask = self._augmentar(image, mask)

        # Convertir a tensor y normalizar imagen
        img_t  = self.normalize(TF.to_tensor(image))
        mask_t = (TF.to_tensor(mask) > 0.5).float()  # Binarizar mascara: 0 o 1

        return img_t, mask_t
