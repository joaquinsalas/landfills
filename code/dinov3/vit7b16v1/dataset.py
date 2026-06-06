import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import albumentations as A
from albumentations.pytorch import ToTensorV2


def read_image(path: str) -> np.ndarray:
    """Lee imagen a numpy (H, W, C) uint8 o float32."""
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff") and HAS_RASTERIO:
        with rasterio.open(path) as src:
            img = src.read()        
            img = np.moveaxis(img, 0, -1)
            if img.dtype != np.float32:
                info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
                if info:
                    img = img.astype(np.float32) / info.max
                else:
                    img = img.astype(np.float32)
        return img
    elif HAS_PIL:
        img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        return img
    else:
        raise ImportError("Instala rasterio o Pillow para leer imágenes.")


def read_mask(path: str) -> np.ndarray:
    """Lee máscara a numpy (H, W) uint8."""
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff") and HAS_RASTERIO:
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.uint8)
    elif HAS_PIL:
        mask = np.array(Image.open(path)).astype(np.uint8)
    else:
        raise ImportError("Instala rasterio o Pillow.")

    # Normaliza 0/255 → 0/1
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    return mask


def get_train_transforms(tile_size: int = 512, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std  = [0.229, 0.224, 0.225]
    return A.Compose([
        A.RandomCrop(tile_size, tile_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
        ], p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std  = [0.229, 0.224, 0.225]
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset principal
# ---------------------------------------------------------------------------

class SatSegDataset(Dataset):
    """
    Parámetros
    ----------
    img_dir   : carpeta con imágenes
    mask_dir  : carpeta con máscaras (mismo nombre de archivo)
    transform : albumentations Compose
    tile_size : tamaño de crop (solo para modo 'train')
    mode      : 'train' | 'val' | 'test'
    extensions: extensiones válidas de imagen
    """

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        transform=None,
        tile_size: int = 512,
        mode: str = "train",
        extensions: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    ):
        self.img_dir   = Path(img_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform
        self.tile_size = tile_size
        self.mode      = mode

        # Listar imágenes
        self.files = sorted([
            f for f in self.img_dir.iterdir()
            if f.suffix.lower() in extensions
        ])
        assert len(self.files) > 0, f"No se encontraron imágenes en {img_dir}"

        # En modo val: pre-calcular todos los tiles
        if mode == "val":
            self.tiles = self._build_val_tiles()
        else:
            self.tiles = None

    def _build_val_tiles(self):
        """Grid de tiles sin solapamiento para validación."""
        tiles = []
        for img_path in self.files:
            img = read_image(str(img_path))
            H, W = img.shape[:2]
            mask_path = self.mask_dir / img_path.name
            for y in range(0, H - self.tile_size + 1, self.tile_size):
                for x in range(0, W - self.tile_size + 1, self.tile_size):
                    tiles.append((str(img_path), str(mask_path), y, x))
        return tiles

    def __len__(self):
        if self.mode == "val":
            return len(self.tiles)
        return len(self.files)

    def __getitem__(self, idx):
        if self.mode == "val":
            img_path, mask_path, y, x = self.tiles[idx]
            img  = read_image(img_path)
            mask = read_mask(mask_path)
            img  = img [y:y+self.tile_size, x:x+self.tile_size]
            mask = mask[y:y+self.tile_size, x:x+self.tile_size]
        else:
            img_path = self.files[idx]
            mask_path = self.mask_dir / img_path.name
            img  = read_image(str(img_path))
            mask = read_mask(str(mask_path))

        # Convertir imagen float32 a uint8 si albumentations lo requiere
        # (depende de las transforms; Normalize espera float)
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img  = augmented["image"]   # tensor (C, H, W)
            mask = augmented["mask"]    # tensor (H, W)

        mask = mask.long()   # CrossEntropy / BCEWithLogitsLoss espera long o float
        return img, mask


# ---------------------------------------------------------------------------
# Factory de DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    img_dir: str,
    mask_dir: str,
    tile_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    val_split: float = 0.15,
    seed: int = 42,
    mean=None,
    std=None,
):
    """
    Divide automáticamente en train/val.
    Devuelve (train_loader, val_loader).
    """
    all_files = sorted([
        f for f in Path(img_dir).iterdir()
        if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    ])
    random.seed(seed)
    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * val_split))
    val_files   = all_files[:n_val]
    train_files = all_files[n_val:]

    # Crear subdirectorios temporales (symlinks lógicos vía lista)
    # → pasamos listas al dataset en lugar de directorios
    # (para simplicidad usamos la misma carpeta y filtramos por lista)

    train_ds = _SubsetDataset(
        files=train_files,
        mask_dir=mask_dir,
        transform=get_train_transforms(tile_size, mean, std),
        tile_size=tile_size,
        mode="train",
    )
    val_ds = _SubsetDataset(
        files=val_files,
        mask_dir=mask_dir,
        transform=get_val_transforms(mean, std),
        tile_size=tile_size,
        mode="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Train: {len(train_ds)} muestras | Val: {len(val_ds)} tiles")
    return train_loader, val_loader


class _SubsetDataset(SatSegDataset):
    """Versión de SatSegDataset que acepta una lista explícita de archivos."""

    def __init__(self, files, mask_dir, transform, tile_size, mode):
        # Inicialización manual (sin llamar a __init__ del padre)
        self.files     = files
        self.mask_dir  = Path(mask_dir)
        self.transform = transform
        self.tile_size = tile_size
        self.mode      = mode
        if mode == "val":
            self.tiles = self._build_val_tiles()
        else:
            self.tiles = None
