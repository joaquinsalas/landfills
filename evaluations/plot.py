import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import config as ini
import numpy as np

def generate_metrics_plot(log_path):
    # Revisa si el archivo de resultados existe, si no, te avisa y se detiene
    if not os.path.exists(log_path):
        print("error")
        return

    # Lee tu archivo CSV con Pandas como si fuera una tabla de Excel
    df = pd.read_csv(log_path)
    epochs = df['epoch']

    # Prepara dos gráficos lado a lado para mostrar la pérdida y la métrica de eficiencia   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Desempeño del Modelo: Segmentación de Rellenos Sanitarios', fontsize=16)

    # Gráfica 1: La Pérdida (Loss) - Buscamos que vaya hacia abajo
    ax1.plot(epochs, df['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, df['val_loss'], label='Val Loss', color='red', linestyle='--', linewidth=2)
    ax1.set_title('Pérdida (Loss)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfica 2: La Puntería (IoU) - Buscamos que vaya hacia arriba
    ax2.plot(epochs, df['train_iou'], label='Train IoU', color='green', linewidth=2)
    ax2.plot(epochs, df['val_iou'], label='Val IoU', color='orange', linestyle='--', linewidth=2)
    ax2.set_title('Métrica de Eficiencia (IoU)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Intersection over Union')
    ax2.set_ylim(0, 1) # Obliga a la gráfica a ir de 0 a 1 (0% a 100%)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Guarda la imagen final en tu carpeta de output
    plot_path = os.path.join(ini.OUTPUT_DIR, "metrics_summary.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"guardada en: {plot_path}")
    plt.show() # Te la muestra en pantalla

    #new(sacar las comparativas)
def visualizar_predicciones(model,val_loader,device,num_imagenes=3):
    """Toma unas cuantas fotos del examen (val_loader) y te muestra qué dibujó la IA."""
    model.eval() # Modo examen, no aprende
    
    # Saca un paquete de imágenes y sus respuestas reales
    images, masks = next(iter(val_loader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        # Aplica la misma lógica de tu IoU.py para dejarlo en blanco y negro puro
        preds = (torch.sigmoid(outputs) > 0.90).float()
        
    # Pasa todo de vuelta al procesador normal para poder dibujarlo
    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()
    
   # Calcula exactamente cuántas filas necesita (si le mandas 2, dibuja 2)
    filas_reales = min(num_imagenes, images.size(0))
    
    fig,axes = plt.subplots(filas_reales, 3, figsize=(12, 4 * filas_reales))
    fig.suptitle('Comparativa: Real vs Inteligencia Artificial', fontsize=16)
    
    # Manejo si solo dibuja una fila
    if filas_reales == 1:
        axes = np.array([axes])

    for i in range(filas_reales):
    
    # Si num_imagenes es 1, subplots devuelve un array 1D, hay que manejarlo
        if num_imagenes == 1:
            axes = np.array([axes])

    for i in range(min(num_imagenes, images.size(0))):
        # 1. IMAGEN ORIGINAL (Satélite)
        img_vis = images[i].permute(1, 2, 0).numpy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
        
        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title("1. Foto Satélite Original")
        axes[i, 0].axis('off')
        
        # 2. MÁSCARA REAL (El sombreado perfecto)
        axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("2. Sombreado Real")
        axes[i, 1].axis('off')
        
        # 3. PREDICCIÓN (Lo que entendió la U-Net)
        axes[i, 2].imshow(preds[i].squeeze(), cmap='gray')
        axes[i, 2].set_title("3. Predicción de la IA")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.show()
#-----
# Si ejecutas este archivo directamente, busca el CSV y llama a la función de arriba
if __name__ == "__main__":
    log_file = os.path.join(ini.OUTPUT_DIR, "training_logs.csv")
    generate_metrics_plot(log_file)