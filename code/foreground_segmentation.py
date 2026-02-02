#!/usr/bin/env python3
"""
Training a Foreground Segmentation Tool with DINOv3
(Modified to use the first 37 matched image/mask pairs from:
 /home/pablo284/.../dinov3/data/{Images,Mask})
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
    print("=== Training a Foreground Segmentation Tool with DINOv3 ===")

    # ---- Output dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"segmentation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # ---- DINOv3 location
    DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"
    DINOV3_LOCATION = os.getenv("DINOV3_LOCATION", DINOV3_GITHUB_LOCATION)

    # ---- Model
    MODEL_NAME = "dinov3_vitl16"
    MODEL_TO_NUM_LAYERS = {
        "dinov3_vits16": 12,
        "dinov3_vits16plus": 12,
        "dinov3_vitb16": 12,
        "dinov3_vitl16": 24,
        "dinov3_vith16plus": 32,
        "dinov3_vit7b16": 40,
    }
    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"DINOv3 location set to {DINOV3_LOCATION}")
    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME} with {n_layers} layers")

    # ---- Load model
    print("\n1. Loading DINOv3 model...")
    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
    ).to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")

    # ---- Data: use first 37 matched pairs in the given path
    print("\n2. Loading training data (first 37 matched pairs)...")
    DATA_DIR = "/home/pablo284/Documents/informs/research/2022.12.21koalas/2025.08.19noseSegmentation/dinov3/data"
    IMAGES_DIR = os.path.join(DATA_DIR, "Image")
    MASKS_DIR  = os.path.join(DATA_DIR, "Mask")

    def list_files(folder):
        exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        return [f for f in os.listdir(folder) if f.endswith(exts)]

    def to_map(folder):
        """basename (no ext) -> absolute path (first one wins if multiple exts)."""
        m = {}
        for name in list_files(folder):
            base = os.path.splitext(name)[0]
            m.setdefault(base, os.path.join(folder, name))
        return m

    img_map  = to_map(IMAGES_DIR)
    mask_map = to_map(MASKS_DIR)

    common = sorted(set(img_map.keys()) & set(mask_map.keys()))
    if not common:
        raise RuntimeError(f"No matching basenames found between {IMAGES_DIR} and {MASKS_DIR}")

    # first 37 in ascending order
    MAX_IMAGES = 37
    chosen = common[:MAX_IMAGES]
    image_paths = [img_map[b] for b in chosen]
    mask_paths  = [mask_map[b] for b in chosen]

    n_images = len(image_paths)
    assert n_images == len(mask_paths), "Image/mask count mismatch after pairing."
    print(f"Using {n_images} matched image/mask pairs.")

    # ---- Constants
    PATCH_SIZE = 16
    IMAGE_SIZE = 768
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # ---- Patch processing
    print("\n3. Setting up patch processing...")
    patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
    patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

    def resize_transform(pil_img, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
        w, h = pil_img.size
        h_patches = image_size // patch_size
        w_patches = int((w * image_size) / (h * patch_size))
        return TF.to_tensor(TF.resize(pil_img, (h_patches * patch_size, w_patches * patch_size)))

    print("Patch processing setup complete")

    # ---- Feature extraction
    features_path = os.path.join(output_dir, f"extracted_features_{timestamp}.npz")
    print("\n4. Extracting features from selected images...")
    xs, ys, image_index = [], [], []

    with torch.inference_mode():
        for i, (ip, mp) in enumerate(tqdm(list(zip(image_paths, mask_paths)), total=n_images, desc="Processing images")):
            # Mask -> L, resize, quantize to patch grid -> flatten labels in [0,1]
            mask_i = Image.open(mp).convert("L")
            mask_resized = resize_transform(mask_i)
            mask_quant = patch_quant_filter(mask_resized).squeeze().view(-1).detach().cpu()
            ys.append(mask_quant)

            # Image -> RGB, resize, normalize, to device
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

    # keep only clear positives/negatives
    idx = (ys < 0.01) | (ys > 0.99)
    xs = xs[idx]
    ys = ys[idx]
    image_index = image_index[idx]

    print("Design matrix:", tuple(xs.shape))
    print("Label vector:", tuple(ys.shape))

    # Save features
    print(f"Saving extracted features to {features_path}...")
    np.savez_compressed(features_path, xs=xs.numpy(), ys=ys.numpy(), image_index=image_index.numpy())
    print("Features saved.")

    # ---- 5-fold CV for C selection (image-wise split)
    print("\n5. 5-fold Cross-validation for model selection...")
    from sklearn.model_selection import KFold

    cs = np.logspace(-7, 0, 8)
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    ap_scores = np.zeros((n_folds, len(cs)))
    dice_scores = np.zeros((n_folds, len(cs)))

    img_idx_np = image_index.numpy()
    uniq_imgs = np.unique(img_idx_np)

    fold = 0
    for tr_imgs, va_imgs in kfold.split(uniq_imgs):
        tr_set = set(uniq_imgs[tr_imgs])
        va_set = set(uniq_imgs[va_imgs])

        tr_sel = np.array([i in tr_set for i in img_idx_np])
        va_sel = np.array([i in va_set for i in img_idx_np])

        Xtr = xs[tr_sel].numpy()
        #ytr = (ys[tr_sel] > 0).astype(int).numpy()
        ytr = (ys[tr_sel] > 0).to(torch.int64).numpy()
        Xva = xs[va_sel].numpy()
        #yva = (ys[va_sel] > 0).astype(int).numpy()
        yva = (ys[va_sel] > 0).to(torch.int64).numpy()

        from sklearn.metrics import precision_recall_curve  # noqa: F401 (import kept for clarity)

        for j, c in enumerate(cs):
            clf = LogisticRegression(random_state=0, C=c, max_iter=10000).fit(Xtr, ytr)
            proba = clf.predict_proba(Xva)[:, 1]
            pred = (proba >= 0.5).astype(int)

            ap = average_precision_score(yva, proba)
            dice = dice_score(yva, pred)

            ap_scores[fold, j] = ap
            dice_scores[fold, j] = dice

        fold += 1

    avg_ap = ap_scores.mean(axis=0)
    avg_dice = dice_scores.mean(axis=0)
    best_idx = int(np.argmax(avg_ap))
    best_c = cs[best_idx]

    print(f"\nBest C: {best_c:.1e}")
    print(f"Avg Precision at best C: {avg_ap[best_idx]:.3f}")
    print(f"Avg Dice at best C: {avg_dice[best_idx]:.3f}")

    # Save CV table
    cv_path = os.path.join(output_dir, f"cv_results_{timestamp}.txt")
    with open(cv_path, "w") as f:
        f.write(f"Cross-Validation Results - {timestamp}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: {n_images} images (first {min(37, n_images)} pairs used)\n")
        f.write(f"Best C: {best_c:.1e}\n")
        f.write(f"Best AP: {avg_ap[best_idx]:.3f}\n")
        f.write(f"Best Dice: {avg_dice[best_idx]:.3f}\n\n")
        f.write("C\t\tAP\tDice\n")
        for c, ap, di in zip(cs, avg_ap, avg_dice):
            f.write(f"{c:.2e}\t{ap:.3f}\t{di:.3f}\n")
    print("CV results saved to:", cv_path)

    # ---- Train final model on all patches
    print("\n6. Training final model...")
    final_clf = LogisticRegression(random_state=0, C=best_c, max_iter=100000, verbose=0)
    #final_clf.fit(xs.numpy(), (ys > 0).astype(int).numpy())
    final_clf.fit(xs.numpy(), (ys > 0).to(torch.int64).numpy())
    print(f"Final classifier trained with C={best_c:.1e}")

    # ---- Test on the first selected image
    print("\n7. Testing on sample image...")
    test_path = image_paths[0]
    test_img = Image.open(test_path).convert("RGB")
    test_resized = resize_transform(test_img)
    test_norm = TF.normalize(test_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    with torch.inference_mode():
        feats = model.get_intermediate_layers(test_norm.unsqueeze(0).to(device), n=range(n_layers), reshape=True, norm=True)
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)

    h_patches, w_patches = [int(d / PATCH_SIZE) for d in test_resized.shape[1:]]
    fg_score = final_clf.predict_proba(x.numpy())[:, 1].reshape(h_patches, w_patches)
    fg_score_mf = signal.medfilt2d(fg_score, kernel_size=3)

    # ---- Save visualizations
    print("\nSaving result images...")
    inp_path = os.path.join(output_dir, f"test_input_{timestamp}.png")
    sc_path  = os.path.join(output_dir, f"test_fg_score_{timestamp}.png")
    mf_path  = os.path.join(output_dir, f"test_fg_score_mf_{timestamp}.png")

    plt.imsave(inp_path, test_resized.permute(1, 2, 0).numpy())
    plt.imsave(sc_path, fg_score, cmap="viridis")
    plt.imsave(mf_path, fg_score_mf, cmap="viridis")

    plt.figure(figsize=(12, 4), dpi=150)
    plt.subplot(1, 3, 1); plt.axis("off"); plt.imshow(test_resized.permute(1, 2, 0)); plt.title("Input")
    plt.subplot(1, 3, 2); plt.axis("off"); plt.imshow(fg_score, cmap="viridis"); plt.title("FG Score"); plt.colorbar(shrink=0.8)
    plt.subplot(1, 3, 3); plt.axis("off"); plt.imshow(fg_score_mf, cmap="viridis"); plt.title("FG Score (MF)"); plt.colorbar(shrink=0.8)
    plt.tight_layout()
    cmp_path = os.path.join(output_dir, f"foreground_segmentation_results_{timestamp}.png")
    plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved images to:", output_dir)

    # ---- Save model
    print("\n8. Saving model...")
    model_path = os.path.join(output_dir, f"fg_classifier_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_clf, f)
    print(f"Classifier saved to {model_path}")

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
