# evaluations/plots.py
# ─────────────────────────────────────────────────────────────────────
# Genera 8 graficas de evaluacion del modelo y las guarda como PNG.
# ─────────────────────────────────────────────────────────────────────
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from config.config       import DEVICE, BATCH_SIZE, IMG_SIZE, SALIDA_METRICAS_PNG
from dataset.dataset     import LandfillDataset
from evaluations.metrics import calcular_todas
from models.unet         import DualEncoderUNet

# Colores del tema
COLOR_BG   = '#f8f9fa'
COLOR_BLUE = '#4c72b0'
COLOR_RED  = '#e74c3c'


def desnormalizar(img_tensor):
    """Convierte imagen normalizada a rango [0,1] para visualizar."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = img_tensor * std + mean
    return torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()


def evaluar_y_graficar(model, img_dir, mask_dir, salida_png=SALIDA_METRICAS_PNG):
    """
    Evalua el modelo sobre el dataset dado y genera graficas de metricas.
    Parametros:
        model      : modelo UNetDINO ya cargado y en modo eval
        img_dir    : carpeta de imagenes
        mask_dir   : carpeta de mascaras
        salida_png : ruta donde guardar el PNG de graficas
    """
    dataset    = LandfillDataset(img_dir, mask_dir, augment=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # Listas para acumular metricas por imagen
    ious, dices, precs, recs, f1s, losses = [], [], [], [], [], []
    ejemplos = []  # Guardar hasta 4 ejemplos visuales

    model.eval()
    for imgs, masks in dataloader:
        imgs  = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        with torch.no_grad():
            preds = model(imgs)

        for i in range(imgs.shape[0]):
            m = calcular_todas(preds[i:i+1], masks[i:i+1])
            ious.append(m['iou'])
            dices.append(m['dice'])
            precs.append(m['precision'])
            recs.append(m['recall'])
            f1s.append(m['f1'])
            losses.append(m['loss'])

            if len(ejemplos) < 4:
                ejemplos.append({
                    'img':  desnormalizar(imgs[i].cpu()),
                    'mask': masks[i, 0].cpu().numpy(),
                    'pred': preds[i, 0].cpu().numpy(),
                    'iou':  m['iou'],
                })

    # Promedios globales
    avg = {
        'IoU':       np.mean(ious),
        'Dice':      np.mean(dices),
        'Precision': np.mean(precs),
        'Recall':    np.mean(recs),
        'F1':        np.mean(f1s),
        'Loss':      np.mean(losses),
    }

    print("\n=== METRICAS GLOBALES ===")
    for k, v in avg.items():
        print(f"  {k:10s}: {v:.4f}")

    # ── GRAFICAS ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(
        'Resultados — DINOv3-vitl16 + ResNet50 + DualDecoder\nSegmentacion de Rellenos Sanitarios',
        fontsize=16, fontweight='bold', color='#1a1a2e', y=0.98
    )

    gs    = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    x_idx = range(len(ious))

    # 1 ── Barras metricas globales ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(COLOR_BG)
    colores = ['#3498db','#2ecc71','#e67e22','#9b59b6','#e74c3c','#1abc9c']
    bars = ax1.bar(list(avg.keys()), list(avg.values()),
                   color=colores, alpha=0.85, zorder=3)
    for bar, val in zip(bars, avg.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.set_title('Metricas Globales Promedio', fontweight='bold')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.grid(axis='y', alpha=0.3, zorder=0)
    ax1.spines[['top','right']].set_visible(False)

    # 2 ── Tabla de metricas ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    tabla_data = [
        [k, f'{v:.4f}', 'Bueno' if v > 0.5 else 'Mejorar']
        for k, v in avg.items()
    ]
    tabla = ax2.table(
        cellText=tabla_data,
        colLabels=['Metrica', 'Valor', 'Estado'],
        loc='center', cellLoc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.8)
    for (r, c), cell in tabla.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1a1a2e')
            cell.set_text_props(color='white', fontweight='bold')
        elif c == 2:
            cell.set_facecolor(
                '#d5f5e3' if tabla_data[r-1][2] == 'Bueno' else '#fdebd0')
        else:
            cell.set_facecolor('#eaf0fb' if r % 2 == 0 else 'white')
        cell.set_edgecolor('#cccccc')
    ax2.set_title('Resumen', fontweight='bold', pad=10)

    # 3 ── Ejemplo visual: imagen, mascara real, prediccion ───────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    ax3.set_title('Imagen Original', fontweight='bold')
    ax3.imshow(ejemplos[0]['img'])

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_title('Mascara Real (Ground Truth)', fontweight='bold')
    ax4.imshow(ejemplos[0]['mask'], cmap='gray')

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    ax5.set_title(f'Prediccion (IoU: {ejemplos[0]["iou"]:.3f})', fontweight='bold')
    ax5.imshow(ejemplos[0]['pred'], cmap='viridis')

    # 4 ── IoU por imagen ──────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.set_facecolor(COLOR_BG)
    ax6.bar(x_idx, ious, color=COLOR_BLUE, alpha=0.7, zorder=3)
    ax6.axhline(np.mean(ious), color=COLOR_RED, linestyle='--',
                linewidth=1.5, label=f'Media: {np.mean(ious):.3f}')
    ax6.set_title('IoU por Imagen', fontweight='bold')
    ax6.set_xlabel('Indice imagen')
    ax6.set_ylabel('IoU')
    ax6.set_ylim(0, 1)
    ax6.legend(fontsize=8)
    ax6.grid(axis='y', alpha=0.3, zorder=0)
    ax6.spines[['top','right']].set_visible(False)

    # 5 ── Precision vs Recall ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.set_facecolor(COLOR_BG)
    sc = ax7.scatter(precs, recs, c=f1s, cmap='RdYlGn',
                     s=60, alpha=0.8, zorder=3,
                     edgecolors='gray', linewidth=0.5,
                     vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax7, label='F1 Score')
    ax7.set_xlabel('Precision')
    ax7.set_ylabel('Recall')
    ax7.set_title('Precision vs Recall (color = F1)', fontweight='bold')
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.plot([0,1],[0,1],'gray', linestyle='--', alpha=0.4, linewidth=1)
    ax7.grid(alpha=0.3, zorder=0)
    ax7.spines[['top','right']].set_visible(False)

    # 6 ── Histograma IoU ──────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.set_facecolor(COLOR_BG)
    ax8.hist(ious, bins=min(10, len(ious)),
             color=COLOR_BLUE, alpha=0.8, edgecolor='white', zorder=3)
    ax8.axvline(np.mean(ious), color=COLOR_RED, linestyle='--',
                linewidth=2, label=f'Media: {np.mean(ious):.3f}')
    ax8.set_title('Distribucion de IoU', fontweight='bold')
    ax8.set_xlabel('IoU')
    ax8.set_ylabel('Frecuencia')
    ax8.legend(fontsize=8)
    ax8.grid(axis='y', alpha=0.3, zorder=0)
    ax8.spines[['top','right']].set_visible(False)

    salida_png = Path(salida_png)
    salida_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(salida_png, dpi=150, bbox_inches='tight', facecolor=COLOR_BG)
    plt.close()
    print(f"Graficas guardadas en: {salida_png}")

    return avg
