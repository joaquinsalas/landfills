import torch
import torch.nn as nn
import torch.optim as optim
import csv
import sys
import os

sys.path.append('/home/emilio/Work/landfills/code/config/')
sys.path.append('/home/emilio/Work/landfills/code/models/')
sys.path.append('/home/emilio/Work/landfills/code/dataset/')
sys.path.append('/home/emilio/Work/landfills/code/evaluations/')

import config as ini
from model_unet import UNet
from loader import prepare_data
from IoU import calculate_iou
from diceloss import DiceLoss
from plot import generate_metrics_plot

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_model():
    if not os.path.exists(ini.OUTPUT_DIR):
        os.makedirs(ini.OUTPUT_DIR)

    scaler = torch.amp.GradScaler('cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    train_loader, val_loader = prepare_data()
    model = UNet(n_channels=3, n_classes=1).to(device)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    log_path = os.path.join(ini.OUTPUT_DIR, "training_logs.csv")

    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_iou', 'val_loss', 'val_iou'])

    for epoch in range(ini.EPOCHS):
        model.train()
        train_loss, train_iou = 0.0, 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
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
                with torch.amp.autocast('cuda'):
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

    torch.save(model.state_dict(), f"{ini.OUTPUT_DIR}/unet_landfills.pth")
    generate_metrics_plot(log_path)
    print(f",odelo y logs guardados en {ini.OUTPUT_DIR}")

if __name__ == "__main__":
    train_model()
