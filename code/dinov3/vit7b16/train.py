"""
Entrenamiento: DINOv3 U-Net para segmentación binaria satelital.

Uso rápido:
    python train.py \
        --ckpt   dinov3_vit7b16_pretrain_sat493m-a6675841.pth \
        --images dataset/images \
        --masks  dataset/masks \
        --epochs 50

Estrategia de fine-tuning en dos fases:
    Fase 1 (primeras N épocas): encoder congelado, solo se entrena el decoder
    Fase 2 (restantes)       : encoder descongelado con LR muy bajo (×10)
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from model   import DinoUNet
from dataset import build_dataloaders


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Dice sobre batch. pred = logits, target = 0/1 long."""
    prob = torch.sigmoid(pred).float()
    tgt  = target.float()
    inter = (prob * tgt).sum()
    union = prob.sum() + tgt.sum()
    return ((2 * inter + eps) / (union + eps)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    prob = (torch.sigmoid(pred) > 0.5).float()
    tgt  = target.float()
    inter = (prob * tgt).sum()
    union = prob.sum() + tgt.sum() - inter
    return ((inter + eps) / (union + eps)).item()


# ---------------------------------------------------------------------------
# Loss combinada: BCE + Dice
# ---------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_w = bce_weight
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        target_f = target.float().unsqueeze(1)   # (B,1,H,W)
        bce_loss  = self.bce(pred, target_f)

        prob  = torch.sigmoid(pred)
        inter = (prob * target_f).sum()
        union = prob.sum() + target_f.sum()
        dice_loss = 1 - (2 * inter + 1) / (union + 1)

        return self.bce_w * bce_loss + (1 - self.bce_w) * dice_loss


# ---------------------------------------------------------------------------
# Entrenamiento / validación por época
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = total_dice = 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(imgs)                       # (B,1,H,W)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_dice += dice_score(logits.squeeze(1), masks)

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, masks)

        total_loss += loss.item()
        total_dice += dice_score(logits.squeeze(1), masks)
        total_iou  += iou_score (logits.squeeze(1), masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True,  help="Ruta al checkpoint DINOv3")
    p.add_argument("--images",      required=True,  help="Directorio de imágenes")
    p.add_argument("--masks",       required=True,  help="Directorio de máscaras")
    p.add_argument("--output",      default="runs/exp1")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--tile_size",   type=int,   default=512)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lr_enc",      type=float, default=1e-5,  help="LR del encoder en fase 2")
    p.add_argument("--unfreeze_at", type=int,   default=10,    help="Época en que se descongela encoder")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--val_split",   type=float, default=0.15)
    p.add_argument("--resume",      default=None, help="Checkpoint para reanudar")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datos ────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        img_dir      = args.images,
        mask_dir     = args.masks,
        tile_size    = args.tile_size,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        val_split    = args.val_split,
    )

    # ── Modelo ───────────────────────────────────────────────────────────
    model = DinoUNet(
        ckpt_path      = args.ckpt,
        num_classes    = 1,
        freeze_encoder = True,   # Fase 1: encoder congelado
    ).to(device)

    # ── Optimizador (solo decoder en fase 1) ─────────────────────────────
    decoder_params = [p for n, p in model.named_parameters()
                      if "encoder" not in n and p.requires_grad]
    optimizer = AdamW(decoder_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()
    criterion = BCEDiceLoss(bce_weight=0.5)

    start_epoch = 0
    best_dice   = 0.0

    # ── Resume ───────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        print(f"Reanudando desde época {start_epoch}")

    # ── Loop de entrenamiento ─────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # ── Fase 2: descongelar encoder ──────────────────────────────────
        if epoch == args.unfreeze_at:
            print(f"\n[Época {epoch}] Descongelando encoder — LR encoder = {args.lr_enc}")
            for p in model.encoder.parameters():
                p.requires_grad_(True)

            # Nuevo optimizador con dos grupos de LR
            optimizer = AdamW([
                {"params": decoder_params,                          "lr": args.lr},
                {"params": list(model.encoder.parameters()),        "lr": args.lr_enc},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
            scaler    = GradScaler()

        # ── Train ─────────────────────────────────────────────────────────
        tr_loss, tr_dice = train_epoch(model, train_loader, optimizer, criterion, scaler, device)

        # ── Val ───────────────────────────────────────────────────────────
        vl_loss, vl_dice, vl_iou = val_epoch(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Época {epoch+1:03d}/{args.epochs} | "
            f"Train loss={tr_loss:.4f} dice={tr_dice:.4f} | "
            f"Val   loss={vl_loss:.4f} dice={vl_dice:.4f} IoU={vl_iou:.4f} | "
            f"{elapsed:.1f}s"
        )

        # ── Checkpoint ───────────────────────────────────────────────────
        state = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "best_dice":  best_dice,
        }
        torch.save(state, out_dir / "last.pth")

        if vl_dice > best_dice:
            best_dice = vl_dice
            torch.save(state, out_dir / "best.pth")
            print(f"  ★ Nuevo mejor Dice: {best_dice:.4f}")

    print(f"\nEntrenamiento finalizado. Mejor Dice: {best_dice:.4f}")
    print(f"Checkpoint guardado en: {out_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
