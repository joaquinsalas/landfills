import config as ini
from dataset import get_dataloader
from transform import get_transforms

def prepare_data():
    # obtiene transformaciones para train y val
    train_tf, val_tf, img_norm = get_transforms(ini.INPUT_SIZE)

    # dataloader de entrenamiento
    train_loader = get_dataloader(
        img_dir=f"{ini.DATA_DIR}/squared/dataset/squared/train/landfills",
        mask_dir=f"{ini.DATA_DIR}/squared/dataset/squared/train/masks",
        batch_size=ini.BATCH_SIZE,
        transform=train_tf,
        img_norm=img_norm,
        num_workers=ini.NUM_WORKERS
    )

    # dataloader de validacion
    val_loader = get_dataloader(
        img_dir=f"{ini.DATA_DIR}/squared/dataset/squared/val/landfills",
        mask_dir=f"{ini.DATA_DIR}/squared/dataset/squared/val/masks",
        batch_size=ini.BATCH_SIZE,
        transform=val_tf,
        img_norm=img_norm,
        num_workers=ini.NUM_WORKERS,
        shuffle=False
    )

    # regresa ambos loaders
    return train_loader, val_loader
