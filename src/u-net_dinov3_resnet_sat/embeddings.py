# embeddings.py
# ─────────────────────────────────────────────────────────────────────
# Genera y guarda embeddings (pooler_output, 1024-d) de cada imagen
# usando DINOv3-vitl16 o DINOv2-large como fallback.
#
# Salida:
#   embeddings.npy         → array (N, 1024) con los vectores
#   embeddings_nombres.txt → un nombre de archivo por linea
#
# Uso:  python embeddings.py
# ─────────────────────────────────────────────────────────────────────
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from config.config import (DEVICE, MODELO_ENCODER, MODELO_FALLBACK,
                            TOKEN_HF, TRAIN_IMG,
                            EMBEDDINGS_NPY, EMBEDDINGS_NAMES,
                            EXTENSIONES)

BATCH_SIZE = 8  # Ajustar segun VRAM disponible

print(f"Dispositivo: {DEVICE}")

# ─── Cargar modelo ────────────────────────────────────────────────────
try:
    processor = AutoImageProcessor.from_pretrained(MODELO_ENCODER, token=TOKEN_HF)
    model     = AutoModel.from_pretrained(MODELO_ENCODER, token=TOKEN_HF)
    nombre_modelo = "DINOv3-vitl16"
    print(f"Modelo: {nombre_modelo} cargado")
except Exception as e:
    print(f"  DINOv3 no disponible: {e}\n  Usando: DINOv2-large")
    processor = AutoImageProcessor.from_pretrained(MODELO_FALLBACK)
    model     = AutoModel.from_pretrained(MODELO_FALLBACK)
    nombre_modelo = "DINOv2-large (fallback)"
    print(f"Modelo: {nombre_modelo} cargado")

model = model.eval().to(DEVICE)

# ─── Listar imagenes ──────────────────────────────────────────────────
imagenes = sorted([
    f for f in os.listdir(TRAIN_IMG)
    if f.lower().endswith(EXTENSIONES)
])

if not imagenes:
    raise FileNotFoundError(f"No se encontraron imagenes en: {TRAIN_IMG}")

print(f"Imagenes encontradas: {len(imagenes)}")

# ─── Generar embeddings en batches ────────────────────────────────────
todos_embeddings = []
todos_nombres    = []

for start in range(0, len(imagenes), BATCH_SIZE):
    batch_nombres = imagenes[start : start + BATCH_SIZE]
    batch_imgs    = [
        Image.open(os.path.join(TRAIN_IMG, n)).convert("RGB")
        for n in batch_nombres
    ]

    # Preprocesar batch completo
    inputs = processor(images=batch_imgs, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # pooler_output: (batch, 1024) resumen global de cada imagen
    embeddings_batch = outputs.pooler_output.cpu().numpy()
    todos_embeddings.extend(embeddings_batch)
    todos_nombres.extend(batch_nombres)

    fin = min(start + BATCH_SIZE, len(imagenes))
    print(f"  [{fin}/{len(imagenes)}] shape batch: {embeddings_batch.shape}")

# ─── Guardar resultados ───────────────────────────────────────────────
embeddings_array = np.array(todos_embeddings)  # (N, 1024)

# Crear la carpeta destino si no existe (EMBEDDINGS_NPY y EMBEDDINGS_NAMES
# se asume que viven en la misma carpeta)
EMBEDDINGS_NPY.parent.mkdir(parents=True, exist_ok=True)

np.save(EMBEDDINGS_NPY, embeddings_array)

with open(EMBEDDINGS_NAMES, 'w', encoding='utf-8') as f:
    for nombre in todos_nombres:
        f.write(nombre + '\n')

print(f"\nModelo usado      : {nombre_modelo}")
print(f"Total embeddings  : {embeddings_array.shape}")
print(f"Guardado en       : {EMBEDDINGS_NPY}")
print(f"Nombres en        : {EMBEDDINGS_NAMES}")
