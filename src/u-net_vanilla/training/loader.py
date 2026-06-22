import config as ini
from dataset import get_dataloader
from transform import get_transforms

def prepare_data():
    train_tf, val_tf, img_norm = get_transforms(ini.INPUT_SIZE)

    train_loader = get_dataloader(
        img_dir=f"{ini.DATA_DIR}/train/landfills",
        mask_dir=f"{ini.DATA_DIR}/train/masks",
        batch_size=ini.BATCH_SIZE,
        transform=train_tf,
        img_norm=img_norm,
        num_workers=ini.NUM_WORKERS
    )

    val_loader = get_dataloader(
        img_dir=f"{ini.DATA_DIR}/val/landfills",
        mask_dir=f"{ini.DATA_DIR}/val/masks",
        batch_size=ini.BATCH_SIZE,
        transform=val_tf,
        img_norm=img_norm,
        num_workers=ini.NUM_WORKERS,
        shuffle=False
    )

    return train_loader, val_loader
