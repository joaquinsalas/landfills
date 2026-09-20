import torch
import torch.nn as nn
import torch.optim as optim
import csv
import json
import os
from pathlib import Path

import config as ini
from evaluations.IoU import calculate_metrics
from evaluations.plot import generate_metrics_plot
from models.model_unet import UNet
from training.diceloss import DiceLoss
from training.loader import prepare_data

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_model():
    Path(ini.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(ini.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    set_seed(ini.SEED)
    run_config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(ini).items()
        if k.isupper() and isinstance(v, (str, int, float, bool, Path, list, tuple))
    }
    with open(ini.RUN_CONFIG_PATH, mode='w') as f:
        json.dump(run_config, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Entrenando en: {device}")

    train_loader, val_loader = prepare_data()
    model = UNet(n_channels=3, n_classes=1).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Parámetros entrenables: {trainable_count:,} / {total_params:,}")

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    optimizer = optim.Adam(trainable_params, lr=ini.LEARNING_RATE)
    
    log_path = Path(ini.TRAINING_LOG_PATH)

    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_loss',
            'train_iou',
            'train_dice',
            'train_precision',
            'train_recall',
            'val_loss',
            'val_iou',
            'val_dice',
            'val_precision',
            'val_recall',
        ])

    best_val_iou = -1.0
    for epoch in range(ini.EPOCHS):
        model.train()
        train_loss = 0.0
        train_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            batch_metrics = calculate_metrics(outputs, masks)
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]

        model.eval()
        val_loss = 0.0
        val_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    v_loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
                
                val_loss += v_loss.item()
                batch_metrics = calculate_metrics(outputs, masks)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]

        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_iou': train_metrics['iou'] / len(train_loader),
            'train_dice': train_metrics['dice'] / len(train_loader),
            'train_precision': train_metrics['precision'] / len(train_loader),
            'train_recall': train_metrics['recall'] / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'val_iou': val_metrics['iou'] / len(val_loader),
            'val_dice': val_metrics['dice'] / len(val_loader),
            'val_precision': val_metrics['precision'] / len(val_loader),
            'val_recall': val_metrics['recall'] / len(val_loader),
        }

        print(f"Epoch [{metrics['epoch']}/{ini.EPOCHS}] "
              f"Train Loss: {metrics['train_loss']:.4f} | "
              f"Val Loss: {metrics['val_loss']:.4f} | "
              f"Val IoU: {metrics['val_iou']:.4f} | "
              f"Val Dice: {metrics['val_dice']:.4f}")

        if metrics['val_iou'] > best_val_iou:
            best_val_iou = metrics['val_iou']
            torch.save(model.state_dict(), ini.MODEL_OUTPUT_PATH)
            print(f"Nuevo mejor modelo guardado: {ini.MODEL_OUTPUT_PATH}")

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics['epoch'], 
                metrics['train_loss'], 
                metrics['train_iou'], 
                metrics['train_dice'],
                metrics['train_precision'],
                metrics['train_recall'],
                metrics['val_loss'], 
                metrics['val_iou'],
                metrics['val_dice'],
                metrics['val_precision'],
                metrics['val_recall'],
            ])

    torch.save(model.state_dict(), ini.LAST_MODEL_OUTPUT_PATH)
    generate_metrics_plot(log_path)
    print(f"Modelo y logs guardados en {ini.OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()
