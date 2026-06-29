from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]  # → parts/code
sys.path.append(str(BASE_DIR / "config"))
sys.path.append(str(BASE_DIR / "dataset"))      # ← dataset.py y transform.py viven aquí
sys.path.append(str(BASE_DIR / "evaluations"))  # ← config.py vive aquí
sys.path.append(str(BASE_DIR / "training"))

import config as ini
from dataset import get_dataloader
from transform import get_transforms

def prepare_data():
    # obtiene transformaciones para train y val
    train_tf, val_tf, img_norm = get_transforms(ini.INPUT_SIZE)

    # dataloader de entrenamiento
    train_loader = get_dataloader(
        img_dir=f"{ini.OUTPUT_DIR}/patches",
        mask_dir=f"{ini.DATA_DIR}/squared/dataset/squared/train/masks",
        batch_size=ini.BATCH_SIZE,
        transform=None,
        img_norm=None,
        num_workers=ini.NUM_WORKERS
    )

    # dataloader de validacion
    val_loader = get_dataloader(
        img_dir=f"{ini.OUTPUT_DIR}/patches",
        mask_dir=f"{ini.DATA_DIR}/squared/dataset/squared/val/masks",
        batch_size=ini.BATCH_SIZE,
        transform=None,
        img_norm=None,
        num_workers=ini.NUM_WORKERS,
        shuffle=True
    )

    # regresa ambos loaders
    return train_loader, val_loader
