import pandas as pd
import matplotlib.pyplot as plt
import os
import config as ini

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

    ax2.plot(epochs, df['train_iou'], label='Train IoU', color='green', linewidth=2)
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
    print(f"guardada en: {plot_path}")
    plt.show()

if __name__ == "__main__":
    log_file = os.path.join(ini.OUTPUT_DIR, "training_logs.csv")
    generate_metrics_plot(log_file)
