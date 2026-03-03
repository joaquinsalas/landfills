#!/usr/bin/env python3
"""
Training a Foreground Segmentation Tool with DINOv2
Proyecto: Identificacion de rellenos sanitarios
"""

import os
import pickle
import datetime

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm


def dice_score(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = (y_true & y_pred).sum()
    tot = y_true.sum() + y_pred.sum()
    return 1.0 if tot == 0 else (2.0 * inter / tot)


def main():
    print("=== Training a Foreground Segmentation Tool with DINOv2 ===")

    # ---- Output dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"segmentation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # ---- DINOv2 desde GitHub de Meta
    DINOV2_LOCATION = "facebookresearch/dinov2"
    MODEL_NAME = "dinov2_vits14"  # version small, ideal para RTX 3050 6GB
    n_layers = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Usando dispositivo: {device}")
    print(f"Modelo: {MODEL_NAME}")

    # ---- Cargar modelo
    print("\n1. Cargando modelo DINOv2...")
    model = torch.hub.load(
        repo_or_dir=DINOV2_LOCATION,
        model=MODEL_NAME,
        source="github",
    ).to(device)
    model.eval()
    print(f"Modelo cargado en {device}")

    # ---- Rutas de tus datos
    IMAGES_DIR = r"C:\proyect_gaby\figures\Image"
    MASKS_DIR  = r"C:\proyect_gaby\figures\Mask"

    # ---- Cargar imagenes
    print("\n2. Cargando imagenes...")

    def list_files(folder):
        exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        return [f for f in os.listdir(folder) if f.endswith(exts)]

    def to_map(folder):
        m = {}
        for name in list_files(folder):
            base = os.path.splitext(name)[0]
            m.setdefault(base, os.path.join(folder, name))
        return m

    img_map  = to_map(IMAGES_DIR)
    mask_map = to_map(MASKS_DIR)

    common = sorted(set(img_map.keys()) & set(mask_map.keys()))
    if not common:
        raise RuntimeError(f"No se encontraron pares coincidentes entre {IMAGES_DIR} y {MASKS_DIR}")

    MAX_IMAGES = 40
    chosen = common[:MAX_IMAGES]
    image_paths = [img_map[b] for b in chosen]
    mask_paths  = [mask_map[b] for b in chosen]

    n_images = len(image_paths)
    print(f"Usando {n_images} pares imagen/mascara.")
    for b in chosen:
        print(f"  Par encontrado: {b}")

    # ---- Constantes
    PATCH_SIZE = 14        # DINOv2 usa parches de 14x14 pixeles
    IMAGE_SIZE = 518       # Multiplo de 14, ideal para tu GPU
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # ---- Filtro de parches
    patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

    def resize_transform(pil_img, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
        w, h = pil_img.size
        h_patches = image_size // patch_size
        w_patches = int((w * image_size) / (h * patch_size))
        return TF.to_tensor(TF.resize(pil_img, (h_patches * patch_size, w_patches * patch_size)))

    # ---- Extraccion de features
    print("\n3. Extrayendo features (esto puede tardar varios minutos)...")
    xs, ys, image_index = [], [], []

    with torch.inference_mode():
        for i, (ip, mp) in enumerate(tqdm(list(zip(image_paths, mask_paths)), total=n_images, desc="Procesando")):
            mask_i = Image.open(mp).convert("L")
            mask_resized = resize_transform(mask_i)
            mask_quant = patch_quant_filter(mask_resized).squeeze().view(-1).detach().cpu()
            ys.append(mask_quant)

            img_i = Image.open(ip).convert("RGB")
            img_resized = resize_transform(img_i)
            img_norm = TF.normalize(img_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)

            feats = model.get_intermediate_layers(img_norm, n=range(n_layers), reshape=True, norm=True)
            dim = feats[-1].shape[1]
            xs.append(feats[-1].squeeze().view(dim, -1).permute(1, 0).detach().cpu())
            image_index.append(i * torch.ones(ys[-1].shape))

    xs = torch.cat(xs)
    ys = torch.cat(ys)
    image_index = torch.cat(image_index)

    # Solo parches claramente positivos o negativos
    idx = (ys < 0.01) | (ys > 0.99)
    xs = xs[idx]
    ys = ys[idx]
    image_index = image_index[idx]

    print("Matriz de features:", tuple(xs.shape))
    print("Vector de etiquetas:", tuple(ys.shape))

    # ---- Validacion cruzada
    print("\n4. Validacion cruzada para seleccionar mejor C...")
    from sklearn.model_selection import KFold

    cs = np.logspace(-7, 0, 8)
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    ap_scores   = np.zeros((n_folds, len(cs)))
    dice_scores_cv = np.zeros((n_folds, len(cs)))

    img_idx_np = image_index.numpy()
    uniq_imgs  = np.unique(img_idx_np)

    fold = 0
    for tr_imgs, va_imgs in kfold.split(uniq_imgs):
        tr_set = set(uniq_imgs[tr_imgs])
        va_set = set(uniq_imgs[va_imgs])

        tr_sel = np.array([i in tr_set for i in img_idx_np])
        va_sel = np.array([i in va_set for i in img_idx_np])

        Xtr = xs[tr_sel].numpy()
        ytr = (ys[tr_sel] > 0).to(torch.int64).numpy()
        Xva = xs[va_sel].numpy()
        yva = (ys[va_sel] > 0).to(torch.int64).numpy()

        for j, c in enumerate(cs):
            clf = LogisticRegression(random_state=0, C=c, max_iter=10000).fit(Xtr, ytr)
            proba = clf.predict_proba(Xva)[:, 1]
            pred  = (proba >= 0.5).astype(int)
            ap_scores[fold, j]      = average_precision_score(yva, proba)
            dice_scores_cv[fold, j] = dice_score(yva, pred)

        fold += 1

    avg_ap   = ap_scores.mean(axis=0)
    avg_dice = dice_scores_cv.mean(axis=0)
    best_idx = int(np.argmax(avg_ap))
    best_c   = cs[best_idx]

    print(f"\nMejor C: {best_c:.1e}")
    print(f"Precision promedio (AP): {avg_ap[best_idx]:.3f}")
    print(f"Dice promedio:           {avg_dice[best_idx]:.3f}")

    # ---- Entrenamiento final
    print("\n5. Entrenando modelo final con todos los datos...")
    final_clf = LogisticRegression(random_state=0, C=best_c, max_iter=100000)
    final_clf.fit(xs.numpy(), (ys > 0).to(torch.int64).numpy())
    print("Modelo final entrenado.")

    # ---- Prediccion en primera imagen
    print("\n6. Generando prediccion en primera imagen de prueba...")
    test_img     = Image.open(image_paths[0]).convert("RGB")
    test_resized = resize_transform(test_img)
    test_norm    = TF.normalize(test_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    with torch.inference_mode():
        feats = model.get_intermediate_layers(test_norm.unsqueeze(0).to(device), n=range(n_layers), reshape=True, norm=True)
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)

    h_patches, w_patches = [int(d / PATCH_SIZE) for d in test_resized.shape[1:]]
    fg_score    = final_clf.predict_proba(x.numpy())[:, 1].reshape(h_patches, w_patches)
    fg_score_mf = signal.medfilt2d(fg_score, kernel_size=3)

    # ---- Guardar resultados
    print("\n7. Guardando resultados...")
    plt.figure(figsize=(12, 4), dpi=150)
    plt.subplot(1, 3, 1); plt.axis("off"); plt.imshow(test_resized.permute(1, 2, 0)); plt.title("Imagen original")
    plt.subplot(1, 3, 2); plt.axis("off"); plt.imshow(fg_score, cmap="viridis");    plt.title("Score relleno"); plt.colorbar(shrink=0.8)
    plt.subplot(1, 3, 3); plt.axis("off"); plt.imshow(fg_score_mf, cmap="viridis"); plt.title("Score suavizado"); plt.colorbar(shrink=0.8)
    plt.tight_layout()

    cmp_path = os.path.join(output_dir, f"resultado_{timestamp}.png")
    plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close()

    model_path = os.path.join(output_dir, f"clasificador_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_clf, f)

    print(f"\nResultados guardados en: {output_dir}")
    print(f"Imagen resultado:        {cmp_path}")
    print(f"Modelo guardado:         {model_path}")
    print("\n=== Listo! ===")


if __name__ == "__main__":
    main()