import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import config as ini
import numpy as np

def generate_metrics_plot(log_path):
    if not os.path.exists(log_path):
        print("error")
        return

    df = pd.read_csv(log_path)
    epochs = df['epoch']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Desempeño del Modelo: Segmentación de Rellenos Sanitarios', fontsize=16)

    ax1.plot(epochs, df['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, df['val_loss'], label='Val Loss', color='red', linestyle='--', linewidth=2)
    ax1.set_title('Pérdida (Loss)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, df['val_iou'], label='Val IoU', color='orange', linestyle='--', linewidth=2)
    ax2.set_title('Métrica de Eficiencia (IoU)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Intersection over Union')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plot_path = os.path.join(ini.OUTPUT_DIR, "metrics_summary.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"guardada en: {plot_path}")


def visualizar_predicciones(model, val_loader, device, num_imagenes=3, save_path=None, images_dir=None):
    model.eval()
    print("  → Obteniendo batch...")
    images, masks = next(iter(val_loader))
    images = images.to(device)

    print("  → Corriendo modelo...")
    with torch.no_grad():
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.99).float()

    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()

    print("  → Creando figura...")
    filas_reales = min(num_imagenes, images.size(0))

    fig, axes = plt.subplots(filas_reales, 3, figsize=(12, 4 * filas_reales))
    fig.suptitle('Comparativa: Real vs Inteligencia Artificial (AlphaEarth)', fontsize=16)

    if filas_reales == 1:
        axes = np.array([axes])

    # Obtiene los nombres de los archivos del val_loader
    nombres = val_loader.dataset.filenames

    for i in range(filas_reales):
        # 1. IMAGEN ORIGINAL .tif
        if images_dir is not None:
            from PIL import Image as PILImage
            tif_path = os.path.join(images_dir, f"{nombres[i]}.tif")
            if os.path.exists(tif_path):
                img_original = PILImage.open(tif_path).convert("RGB")
                img_original = np.array(img_original)
                axes[i, 0].imshow(img_original)
            else:
                axes[i, 0].imshow(np.zeros((512, 512, 3)))
        else:
            # Fallback: falso color con bandas 0-2
            img_vis = images[i][[0, 1, 2], :, :].permute(1, 2, 0).numpy()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-6)
            axes[i, 0].imshow(img_vis)

        axes[i, 0].set_title("1. Foto Satélite Original")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("2. Sombreado Real")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(preds[i].squeeze(), cmap='gray')
        axes[i, 2].set_title("3. Predicción de la IA")
        axes[i, 2].axis('off')

    plt.tight_layout()

    print(f"  → Guardando en {save_path}...")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print("  → ¡Guardado!")
    plt.close(fig)


if __name__ == "__main__":
    log_file = os.path.join(ini.OUTPUT_DIR, "training_logs.csv")
    generate_metrics_plot(log_file)