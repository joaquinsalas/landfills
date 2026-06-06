import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from model   import DinoUNet
from dataset import build_dataloaders

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
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

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_w = bce_weight
        self.bce   = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        target_f = target.float().unsqueeze(1)
        bce_loss  = self.bce(pred, target_f)

        prob  = torch.sigmoid(pred)
        inter = (prob * target_f).sum()
        union = prob.sum() + target_f.sum()
        dice_loss = 1 - (2 * inter + 1) / (union + 1)

        return self.bce_w * bce_loss + (1 - self.bce_w) * dice_loss

def train_epoch(model, loader, optimizer, criterion, scaler, device, accumulation_steps=4):
    model.train()
    total_loss = total_dice = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    use_bf16 = torch.cuda.is_bf16_supported()

    for i, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        
        # AMP dinámico dependiendo del hardware (bfloat16 no requiere escalado de gradiente)
        with autocast(enabled=True, dtype=torch.bfloat16 if use_bf16 else torch.float16):
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss   = loss / accumulation_steps

        if use_bf16:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            if not use_bf16:
                scaler.unscale_(optimizer)
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if use_bf16:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
                
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        total_dice += dice_score(logits.squeeze(1), masks)

    n = len(loader)
    return total_loss / n, total_dice / n

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    use_bf16 = torch.cuda.is_bf16_supported()

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with autocast(enabled=True, dtype=torch.bfloat16 if use_bf16 else torch.float16):
            logits = model(imgs)
            loss   = criterion(logits, masks)

        total_loss += loss.item()
        total_dice += dice_score(logits.squeeze(1), masks)
        total_iou  += iou_score (logits.squeeze(1), masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--images",      required=True)
    p.add_argument("--masks",       required=True)
    p.add_argument("--output",      default="runs/exp1")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--tile_size",   type=int,   default=512)
    p.add_argument("--batch_size",  type=int,   default=2, help="Reduce a 1 o 2 si persiste OOM")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--lr_enc",      type=float, default=1e-5)
    p.add_argument("--unfreeze_at", type=int,   default=10)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--val_split",   type=float, default=0.15)
    p.add_argument("--resume",      default=None)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo detectado: {device}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(
        img_dir      = args.images,
        mask_dir     = args.masks,
        tile_size    = args.tile_size,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        val_split    = args.val_split,
    )

    model = DinoUNet(ckpt_path=args.ckpt, num_classes=1, freeze_encoder=True).to(device)

    decoder_params = [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad]
    optimizer = AdamW(decoder_params, lr=args.lr, weight_decay=1e-4, fused=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()
    criterion = BCEDiceLoss(bce_weight=0.5)

    start_epoch = 0
    best_dice   = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        print(f"Reanudando desde época {start_epoch}")

    # Calcular pasos necesarios para simular un batch size efectivo de 8
    accum_steps = max(1, 8 // args.batch_size)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        if epoch == args.unfreeze_at:
            print(f"\n[Época {epoch}] Descongelando bloques superiores del ViT para ajuste fino...")
            
            # Descongelar únicamente bloques profundos (36 a 39) para no desbordar los estados del optimizador
            for name, param in model.encoder.vit.named_parameters():
                if any(f"blocks.{j}." in name for j in range(36, 40)) or "norm" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

            encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
            
            optimizer = AdamW([
                {"params": decoder_params, "lr": args.lr},
                {"params": encoder_params,  "lr": args.lr_enc},
            ], weight_decay=1e-4, fused=True)
            
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)

        tr_loss, tr_dice = train_epoch(model, train_loader, optimizer, criterion, scaler, device, accumulation_steps=accum_steps)
        vl_loss, vl_dice, vl_iou = val_epoch(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Época {epoch+1:03d}/{args.epochs} | "
            f"Train loss={tr_loss:.4f} dice={tr_dice:.4f} | "
            f"Val   loss={vl_loss:.4f} dice={vl_dice:.4f} IoU={vl_iou:.4f} | "
            f"{elapsed:.1f}s"
        )

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

if __name__ == "__main__":
    main()
