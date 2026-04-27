from pathlib import Path
import sys

from transform import get_transforms
from dataset import get_dataloader
import torch

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
if str(CONFIG_DIR) not in sys.path:
    sys.path.append(str(CONFIG_DIR))

import config as ini

train_tf, val_tf, img_norm = get_transforms(ini.INPUT_SIZE)

TRAIN_IMG_PATH = f"{ini.DATA_DIR}/squared/train/landfills"
TRAIN_MASK_PATH = f"{ini.DATA_DIR}/squared/train/masks"

train_loader = get_dataloader(
    img_dir=TRAIN_IMG_PATH,
    mask_dir=TRAIN_MASK_PATH,
    batch_size=ini.BATCH_SIZE,
    transform=train_tf,
    img_norm=img_norm,
    num_workers=ini.NUM_WORKERS
)

images, masks = next(iter(train_loader))

print(f"Dimensiones de Imágenes: {images.shape}")
print(f"Dimensiones de Máscaras: {masks.shape}")

print(f"Tipo de dato Imagen: {images.dtype} | Rango: [{images.min():.2f}, {images.max():.2f}]")
print(f"Tipo de dato Máscara: {masks.dtype} | Rango: [{masks.min():.2f}, {masks.max():.2f}]")

valores_unicos = torch.unique(masks)
print(f"Valores únicos encontrados en la mascara: {valores_unicos}")

if len(valores_unicos) > 2:
    print("Alerta: La máscara tiene más de 2 valores. Revisa la interpolación al redimensionar.")
