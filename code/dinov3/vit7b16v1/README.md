# DINOv3 ViT-B/16 → U-Net — Segmentación Binaria Satelital

## Arquitectura

```
Imagen (B, 3, H, W)
    │
    ▼
┌─────────────────────────────────────┐
│  DINOv3 ViT-B/16  (patch=16)        │
│  Capas intermedias: 2, 5, 8, 11     │
│  Embed dim: 768                     │
└──────────┬──────────────────────────┘
           │ 4 × (B, h_p·w_p, 768)
           ▼
    TokenToMap (Linear + reshape)
           │
    f0 (capa 2)  → (B,  32, H/16, W/16)
    f1 (capa 5)  → (B,  64, H/16, W/16)
    f2 (capa 8)  → (B, 128, H/16, W/16)
    f3 (capa 11) → (B, 256, H/16, W/16)  ← bottleneck
           │
           ▼
    Decoder U-Net (4 etapas ×2 cada una)
    dec0: f3 + skip f2 → (B, 128, H/8,  W/8)
    dec1: d0 + skip f1 → (B,  64, H/4,  W/4)
    dec2: d1 + skip f0 → (B,  32, H/2,  W/2)
    dec3: d2            → (B,  16, H,    W)     ← resolución original
           │
           ▼
    Conv 1×1 → (B, 1, H, W)   logit binario
```

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura del dataset

```
dataset/
├── images/
│   ├── img_001.tif
│   └── img_002.tif
└── masks/
    ├── img_001.tif   # 0 = fondo, 1 = clase positiva
    └── img_002.tif
```

Las máscaras pueden ser 0/255 (se normalizan a 0/1 automáticamente).

## Entrenamiento

```bash
python train.py \
    --ckpt   dinov3_vit7b16_pretrain_sat493m-a6675841.pth \
    --images dataset/images \
    --masks  dataset/masks \
    --epochs 50 \
    --tile_size 512 \
    --batch_size 4 \
    --lr 1e-4 \
    --unfreeze_at 10    # época en que se descongela el encoder
```

### Estrategia de fine-tuning en dos fases

| Fase | Épocas | Encoder | Decoder | LR decoder | LR encoder |
|------|--------|---------|---------|------------|------------|
| 1    | 0–9    | ❄ congelado | ✓ | 1e-4 | — |
| 2    | 10–50  | ✓ descongelado | ✓ | 1e-4 | 1e-5 |

Esto evita destruir los pesos preentrenados en las primeras épocas.

## Inferencia sobre imágenes grandes

```bash
python infer.py \
    --ckpt       runs/exp1/best.pth \
    --model_ckpt dinov3_vit7b16_pretrain_sat493m-a6675841.pth \
    --input      imagen_10000x10000.tif \
    --output     prediccion.tif \
    --tile_size  512 \
    --overlap    128 \
    --threshold  0.5
```

El overlap con ventana de Hann elimina artefactos en los bordes de cada tile.

## Verificación rápida del modelo

```bash
python model.py dinov3_vit7b16_pretrain_sat493m-a6675841.pth
# Output:
# Input : torch.Size([1, 3, 512, 512])
# Output: torch.Size([1, 1, 512, 512])
# ✓ Forward pass OK
```

## Notas importantes

- **Canales de entrada**: el modelo espera 3 canales. Si tu SAR es 1 canal (amplitud),
  duplícalo: `img = np.repeat(img, 3, axis=-1)`.
  Si tienes SAR dual-pol (2 canales), añade un canal sintético o ajusta `patch_embed`.

- **tile_size**: debe ser múltiplo de 16 (patch_size del ViT).
  Recomendado: 512 o 1024 según VRAM disponible.

- **Normalización**: por defecto usa media/std de ImageNet. Si tu SAR tiene
  estadísticas muy distintas, calcula las tuyas y pásalas a `build_dataloaders`.

- **Loss**: BCEDiceLoss (50% BCE + 50% Dice). Para datasets muy desbalanceados
  considera aumentar el peso del Dice o usar Focal Loss.
