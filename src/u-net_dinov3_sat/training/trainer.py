import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
from pathlib import Path

import config as ini
from evaluations.IoU import calculate_iou
from evaluations.plot import generate_metrics_plot
from models.model_unet import UNet
from training.diceloss import DiceLoss
from training.loader import prepare_data

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_model():
    Path(ini.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(ini.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Entrenando en: {device}")

    train_loader, val_loader = prepare_data()
    model = UNet(n_channels=3, n_classes=1).to(device)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=ini.LEARNING_RATE)
    
    log_path = Path(ini.TRAINING_LOG_PATH)

    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_iou', 'val_loss', 'val_iou'])

    for epoch in range(ini.EPOCHS):
        model.train()
        train_loss, train_iou = 0.0, 0.0
        
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
            train_iou += calculate_iou(outputs, masks)

        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    v_loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
                
                val_loss += v_loss.item()
                val_iou += calculate_iou(outputs, masks)

        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_iou': train_iou / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'val_iou': val_iou / len(val_loader)
        }

        print(f"Epoch [{metrics['epoch']}/{ini.EPOCHS}] "
              f"Train Loss: {metrics['train_loss']:.4f} | "
              f"Val Loss: {metrics['val_loss']:.4f} | "
              f"Val IoU: {metrics['val_iou']:.4f}")

        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics['epoch'], 
                metrics['train_loss'], 
                metrics['train_iou'], 
                metrics['val_loss'], 
                metrics['val_iou']
            ])

    torch.save(model.state_dict(), ini.MODEL_OUTPUT_PATH)
    generate_metrics_plot(log_path)
    print(f"Modelo y logs guardados en {ini.OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()
