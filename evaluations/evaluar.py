import torch
import numpy as np
import sys
import os


# 1. CONTROL DE RUTAS DINÁMICAS (MÉTODO TU TRAINER)

_f_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_f_dir) != "parts" and _f_dir != os.path.dirname(_f_dir):
    _f_dir = os.path.dirname(_f_dir)
PART_DIR = _f_dir

# Forzamos los accesos del sistema a tus carpetas internas de código
sys.path.append(os.path.join(PART_DIR, "code", "config"))
sys.path.append(os.path.join(PART_DIR, "code", "models"))
sys.path.append(os.path.join(PART_DIR, "code", "dataset"))
sys.path.append(os.path.join(PART_DIR, "code")) 

# IMPORTS COMPLETAMENTE ALINEADOS A TU PROYECTO ACTUAL
import config as ini
from Res_Unet34 import ResUNet34         # Tu arquitectura ResNet34
from dataset import get_dataloader       # Tu cargador real de dataset.py
from transform import get_transforms     # Tus transformaciones de transform.py
from IoU import calculate_iou           # Tu métrica de IoU.py


# 2. ALGORITMO CALIBRADOR DE UMBRALES

def encontrar_umbral_optimo(model, val_loader, device):
    model.eval() 
    umbrales_a_probar = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    mejor_iou = -1.0  
    mejor_umbral = 0.5
    
    print("\n=== [INICIANDO CALIBRACIÓN DE LA RESUNET-34] ===")
    print("Evaluando comportamiento de píxeles en el set de validación...\n")
    
    with torch.no_grad():
        for threshold in umbrales_a_probar:
            iou_acumulado = 0.0
            total_batches = 0
            
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                iou_batch = calculate_iou(outputs, masks, threshold=threshold)
                iou_acumulado += iou_batch
                total_batches += 1
            
            iou_promedio = iou_acumulado / total_batches
            print(f" -> Probando Umbral: {threshold:.2f} | IoU Promedio: {iou_promedio:.4f}")
            
            if iou_promedio > mejor_iou:
                mejor_iou = iou_promedio
                mejor_umbral = threshold
                
    print("\n=======================================================")
    print(" ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
    print(f" Configuración más óptima recomendada: {mejor_umbral}")
    print(f" IoU Máximo alcanzado con este umbral: {mejor_iou:.4f}")
    print("=======================================================\n")
    return mejor_umbral


# 3. BLOQUE PRINCIPAL DE EJECUCIÓN (CORREGIDO PARA TU RUTA ANIDADA)

if __name__ == "__main__":
    # Detección de hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ejecutando evaluación en el dispositivo: {device}")
    
    # 1. Cargamos transformaciones usando tu tamaño estándar (256)
    _, val_transform, img_norm = get_transforms(ini.INPUT_SIZE)
    
    # 2. DEFINICIÓN DE RUTAS 
    val_img_path = os.path.join(ini.DATA_DIR, "squared", "dataset", "squared", "val", "landfills")
    val_mask_path = os.path.join(ini.DATA_DIR, "squared", "dataset", "squared", "val", "masks")
    
    print(f"Buscando imágenes en: {val_img_path}")
    print(f"Buscando máscaras en: {val_mask_path}")
    
    # 3. Construimos el val_loader real
    val_loader = get_dataloader(
        img_dir=val_img_path,
        mask_dir=val_mask_path,
        batch_size=ini.BATCH_SIZE,
        transform=val_transform,
        img_norm=img_norm,
        shuffle=False, 
        num_workers=ini.NUM_WORKERS
    )
    
    # 4. Instanciamos la red neuronal ResUNet34
    print("Instanciando red neuronal ResUNet34...")
    model = ResUNet34()
    
    # 5. Apuntamos a tu carpeta oficial de salidas para buscar los pesos
    ruta_pesos = os.path.join(ini.OUTPUT_DIR, "resunet_landfills_best.pth")
    
    if os.path.exists(ruta_pesos):
        print(f"Cargando pesos guardados desde: {ruta_pesos}")
        model.load_state_dict(torch.load(ruta_pesos, map_location=device))
        model = model.to(device)
        
        print("Iniciando la búsqueda del punto dulce...")
        umbral_ganador = encontrar_umbral_optimo(model, val_loader, device)
    else:
        print(f"\n❌ ERROR: No encontré tus pesos entrenados en la ruta '{ruta_pesos}'.")
        print(f"Asegúrate de haber completado el entrenamiento de la ResUNet34 para que el archivo exista en: {ini.OUTPUT_DIR}")