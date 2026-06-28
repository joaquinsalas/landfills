# training/train.py
# ─────────────────────────────────────────────────────────────────────
# Loop de entrenamiento principal para DualEncoderUNet (DINOv3 + ResNet50).
# Usa BCEDiceLoss, AdamW con LR diferente para DINOv3 y el resto del
# modelo (ResNet50 + decoder), CosineAnnealingLR y descongelado
# automatico de DINOv3 en epoch UNFREEZE_EPOCH.
# ─────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config  import (DEVICE, EPOCHS, BATCH_SIZE,
                             LR_DECODER, LR_ENCODER, WEIGHT_DECAY,
                             UNFREEZE_EPOCH, UNFREEZE_LAYERS,
                             TRAIN_IMG, TRAIN_MASK, VAL_IMG, VAL_MASK,
                             MODELO_PTH)
from dataset.dataset import LandfillDataset
from models.unet     import DualEncoderUNet


class BCEDiceLoss(nn.Module):
    """
    Funcion de perdida combinada: 50% BCE + 50% Dice.
    BCE penaliza errores pixel a pixel.
    Dice es mas sensible a clases pequeñas (rellenos pequeños).
    """
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # BCE Loss
        bce = nn.BCELoss()(pred, target)

        # Dice Loss
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / \
               (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 0.5 * bce + 0.5 * dice  # Balance 50/50


def calcular_iou(preds, masks):
    """Calcula IoU promedio del batch rapidamente."""
    p_bin = (preds > 0.5).float()
    inter = (p_bin * masks).sum(dim=(1, 2, 3))
    union = (p_bin + masks).clamp(0, 1).sum(dim=(1, 2, 3))
    return (inter / (union + 1e-8)).mean().item()


def entrenar():
    print(f"Usando: {DEVICE}")

    # Datasets con y sin augmentation
    train_dataset = LandfillDataset(TRAIN_IMG, TRAIN_MASK, augment=True)
    val_dataset   = LandfillDataset(VAL_IMG,   VAL_MASK,   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)} imagenes | Val: {len(val_dataset)} imagenes")

    # Modelo y funcion de perdida
    model     = DualEncoderUNet().to(DEVICE)
    criterion = BCEDiceLoss()

    # Optimizador con LR diferente segun el bloque:
    #   - 'dino'  -> encoder DINOv3 (congelado al inicio, LR_ENCODER al descongelar)
    #   - resto   -> ResNet50 + proyecciones + fusion + decoder (siempre entrenable, LR_DECODER)
    decoder_params = [p for n, p in model.named_parameters()
                      if 'dino' not in n]
    encoder_params = [p for n, p in model.named_parameters()
                      if 'dino' in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': decoder_params, 'lr': LR_DECODER},
        {'params': encoder_params, 'lr': LR_ENCODER}
    ], weight_decay=WEIGHT_DECAY)

    # Scheduler coseno: reduce LR suavemente a lo largo del entrenamiento
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    # Loop de entrenamiento
    for epoch in range(EPOCHS):

        # Descongelar encoder en la epoca indicada
        if epoch == UNFREEZE_EPOCH:
            model.descongelar_encoder(UNFREEZE_LAYERS)
            # Recrear optimizador para incluir los nuevos parametros descongelados
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR_DECODER,
                weight_decay=WEIGHT_DECAY)

        # ── Entrenamiento ─────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)

            optimizer.zero_grad()          # Limpiar gradientes
            preds = model(images)          # Forward pass
            loss  = criterion(preds, masks)  # Calcular perdida
            loss.backward()                # Backpropagation
            optimizer.step()               # Actualizar pesos
            train_loss += loss.item()

        # ── Validacion ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0

        with torch.no_grad():
            for v_imgs, v_masks in val_loader:
                v_imgs  = v_imgs.to(DEVICE)
                v_masks = v_masks.to(DEVICE)
                v_preds = model(v_imgs)
                val_loss += criterion(v_preds, v_masks).item()
                val_iou  += calcular_iou(v_preds, v_masks)

        scheduler.step()

        # Imprimir metricas de la epoca
        n_train = len(train_loader)
        n_val   = len(val_loader)
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss/n_train:.4f} | "
              f"Val Loss: {val_loss/n_val:.4f} | "
              f"Val IoU: {val_iou/n_val:.4f}")

    # Guardar modelo entrenado (crea la carpeta destino si no existe)
    MODELO_PTH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELO_PTH)
    print(f"Modelo guardado en: {MODELO_PTH}")


if __name__ == "__main__":
    entrenar()
