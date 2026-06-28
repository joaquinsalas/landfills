# test.py
# ─────────────────────────────────────────────────────────────────────
# Inferencia y visualizacion del modelo entrenado.
# Carga el modelo guardado, evalua sobre val o train y genera graficas.
#
# Uso:  python test.py
# ─────────────────────────────────────────────────────────────────────
import os
import torch

from config.config       import (DEVICE, MODELO_PTH,
                                 VAL_IMG, VAL_MASK,
                                 TRAIN_IMG, TRAIN_MASK,
                                 SALIDA_METRICAS_PNG)
from models.unet         import DualEncoderUNet
from evaluations.plots   import evaluar_y_graficar

print(f"Dispositivo: {DEVICE}")

# ─── Cargar modelo ────────────────────────────────────────────────────
if not os.path.exists(MODELO_PTH):
    print(f"ERROR: No se encuentra el modelo en {MODELO_PTH}")
    print("Ejecuta primero: python training/train.py")
    exit()

model = DualEncoderUNet().to(DEVICE)
model.load_state_dict(
    torch.load(MODELO_PTH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Modelo cargado desde: {MODELO_PTH}")

# ─── Seleccionar carpeta de evaluacion ───────────────────────────────
# Si existe la carpeta val, usarla; si no, usar train como demo
if os.path.exists(VAL_IMG):
    img_dir  = VAL_IMG
    mask_dir = VAL_MASK
    print("Evaluando sobre: val")
else:
    img_dir  = TRAIN_IMG
    mask_dir = TRAIN_MASK
    print("Carpeta val no encontrada, usando train para demo")

# ─── Evaluar y generar graficas ───────────────────────────────────────
avg = evaluar_y_graficar(
    model     = model,
    img_dir   = img_dir,
    mask_dir  = mask_dir,
    salida_png= SALIDA_METRICAS_PNG
)
