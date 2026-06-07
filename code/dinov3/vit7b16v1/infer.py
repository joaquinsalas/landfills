"""
Inferencia con tiling solapado para imágenes SAR/satelitales grandes.

El tiling con overlap evita artefactos en los bordes de cada tile.
Se usa una ventana de Hann para mezclar las predicciones.

Uso:
    python infer.py \
        --ckpt  runs/exp1/best.pth \
        --model_ckpt dinov3_vit7b16_pretrain_sat493m-a6675841.pth \
        --input  imagen_grande.tif \
        --output prediccion.tif \
        --tile_size 512 \
        --overlap   128 \
        --threshold 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import DinoUNet

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Ventana de Hann 2D para blending
# ---------------------------------------------------------------------------

def hann_window_2d(size: int) -> np.ndarray:
    w1d = np.hanning(size).astype(np.float32)
    return np.outer(w1d, w1d)


# ---------------------------------------------------------------------------
# Lectura / escritura agnóstica al formato
# ---------------------------------------------------------------------------

def load_image(path: str):
    """Devuelve (H, W, C) float32 normalizado a [0,1] + profile rasterio si existe."""
    profile = None
    if Path(path).suffix.lower() in (".tif", ".tiff") and HAS_RASTERIO:
        with rasterio.open(path) as src:
            profile = src.profile
            img = src.read()                      # (C, H, W)
            img = np.moveaxis(img, 0, -1).astype(np.float32)
            if img.max() > 1.0:
                img = img / np.iinfo(np.uint8).max if img.max() <= 255 else img / img.max()
    elif HAS_PIL:
        img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    else:
        raise ImportError("Instala rasterio o Pillow.")
    return img, profile


def save_mask(pred: np.ndarray, path: str, profile=None):
    """pred: (H, W) float32 en [0,1]."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff") and HAS_RASTERIO and profile is not None:
        profile.update(dtype=rasterio.uint8, count=1, compress="lzw")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write((pred * 255).astype(np.uint8)[np.newaxis])
    elif HAS_PIL:
        Image.fromarray((pred * 255).astype(np.uint8)).save(str(path))
    else:
        raise ImportError("Instala rasterio o Pillow.")


# ---------------------------------------------------------------------------
# Inferencia con tiling
# ---------------------------------------------------------------------------

def predict_tiled(
    model: torch.nn.Module,
    img: np.ndarray,                # (H, W, C) float32
    tile_size: int = 512,
    overlap: int = 128,
    threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
    mean=(0.485, 0.456, 0.406),
    std =(0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Devuelve máscara probabilística (H, W) float32.
    """
    H, W, C = img.shape
    stride   = tile_size - overlap

    # Acumuladores
    prob_map   = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    window     = hann_window_2d(tile_size)

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)

    model.eval()
    with torch.no_grad():
        y = 0
        while y < H:
            y_end = min(y + tile_size, H)
            y_start = y_end - tile_size
            if y_start < 0:
                y_start = 0; y_end = tile_size

            x = 0
            while x < W:
                x_end = min(x + tile_size, W)
                x_start = x_end - tile_size
                if x_start < 0:
                    x_start = 0; x_end = tile_size

                tile = img[y_start:y_end, x_start:x_end]   # (T, T, C)

                # Normalizar
                tile_norm = (tile - mean_arr) / std_arr
                tensor = torch.from_numpy(tile_norm).permute(2, 0, 1).unsqueeze(0).to(device)

                # Predecir
                logit = model(tensor)                         # (1, 1, T, T)
                prob  = torch.sigmoid(logit).squeeze().cpu().numpy()  # (T, T)

                # Blend con ventana de Hann
                prob_map  [y_start:y_end, x_start:x_end] += prob   * window
                weight_map[y_start:y_end, x_start:x_end] += window

                x += stride
                if x_end == W:
                    break

            y += stride
            if y_end == H:
                break

    # Normalizar
    prob_map /= np.maximum(weight_map, 1e-6)
    return prob_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True, help="Checkpoint del modelo entrenado (runs/.../best.pth)")
    p.add_argument("--model_ckpt", required=True, help="Checkpoint DINOv3 original")
    p.add_argument("--input",      required=True, help="Imagen de entrada")
    p.add_argument("--output",     required=True, help="Ruta de salida de la máscara")
    p.add_argument("--tile_size",  type=int,   default=512)
    p.add_argument("--overlap",    type=int,   default=128)
    p.add_argument("--threshold",  type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ── Modelo ────────────────────────────────────────────────────────────
    model = DinoUNet(ckpt_path=args.model_ckpt, num_classes=1).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    print(f"Modelo cargado desde {args.ckpt}")

    # ── Imagen ───────────────────────────────────────────────────────────
    img, profile = load_image(args.input)
    print(f"Imagen: {img.shape}  ({img.dtype})")

    # ── Inferencia ────────────────────────────────────────────────────────
    prob = predict_tiled(
        model     = model,
        img       = img,
        tile_size = args.tile_size,
        overlap   = args.overlap,
        threshold = args.threshold,
        device    = device,
    )

    # ── Guardar ──────────────────────────────────────────────────────────
    save_mask(prob, args.output, profile)
    n_pos = (prob > args.threshold).sum()
    pct   = 100 * n_pos / prob.size
    print(f"Predicción guardada en {args.output}")
    print(f"Píxeles positivos: {n_pos:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
