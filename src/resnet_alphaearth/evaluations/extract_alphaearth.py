import os
import ee
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import rasterio
import numpy as np
import torch
import requests
import io
import config as ini

# 1. Api de gee
print("Autenticando con Google Earth Engine")
ee.Authenticate(auth_mode='notebook')
ee.Initialize(project='rellenos-sanitarios-ae')

# 2. Rutas
IMAGES_DIR = os.path.join(ini.DATA_DIR, "squared", "dataset", "squared", "train", "landfills")
OUTPUT_ALPHA_DIR = os.path.join(ini.OUTPUT_DIR, "patches")

if not os.path.exists(OUTPUT_ALPHA_DIR):
    os.makedirs(OUTPUT_ALPHA_DIR)

# 3. CONEXIÓN AL MODELO ALPHAEARTH
print("Conectando con AlphaEarth de Google DeepMind")
alphaearth_collection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
                            .filterDate('2024-01-01', '2025-01-01')

band_names = [f'A{str(i).zfill(2)}' for i in range(64)]
DELTA = 0.005  # Aproximadamente 500 metros de radio alrededor del punto central

# 4. ESCANEO DE ARCHIVOS
filenames = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.tif')])
print(f"Se encontraron {len(filenames)} imágenes GeoTIFF. Iniciando escaneo\n")

for fname in filenames:
    img_path = os.path.join(IMAGES_DIR, fname)
    name_without_ext = os.path.splitext(fname)[0]

    try:
        # Extraer coordenadas del .tif
        with rasterio.open(img_path) as src:
            lon, lat = src.lnglat()

        print(f"Procesando {fname} → Lat {lat:.4f}, Lon {lon:.4f}")

        # Crear AOI
        aoi = ee.Geometry.Rectangle([
            lon - DELTA, lat - DELTA,
            lon + DELTA, lat + DELTA
        ])

        # Obtener imagen de AlphaEarth
        embedding_image = alphaearth_collection.filterBounds(aoi).mosaic().clip(aoi)

        # Descargar como array NumPy directamente
        url = embedding_image.getDownloadURL({
            'bands': band_names,
            'region': aoi,
            'scale': 10,
            'format': 'NPY'
        })

        response = requests.get(url)
        response.raise_for_status()

        # Cargar y reordenar dimensiones: (H, W, 64) → (64, H, W)
        matriz_np = np.load(io.BytesIO(response.content))
        capas_64 = [matriz_np[banda].astype(np.float32) for banda in band_names]
        matriz_np = np.stack(capas_64, axis=0)  # → (64, H, W)

        # Convertir a tensor y recortar/pad a exactamente [64, 512, 512]
        tensor_alpha = torch.from_numpy(matriz_np).float()

        # Recortar si es más grande
        tensor_alpha = tensor_alpha[:, :512, :512]

        # Pad si es más pequeño
        _, h, w = tensor_alpha.shape
        if h < 512 or w < 512:
            pad_h = 512 - h
            pad_w = 512 - w
            tensor_alpha = torch.nn.functional.pad(tensor_alpha, (0, pad_w, 0, pad_h))

        # Guardar tensor
        save_path = os.path.join(OUTPUT_ALPHA_DIR, f"{name_without_ext}.pt")
        torch.save(tensor_alpha, save_path)
        print(f"  Guardado: {tensor_alpha.shape} → {save_path}\n")

    except Exception as e:
        print(f"  ✗ Saltando {fname}: {e}\n")

print(f"¡Proceso completado! Tensores en: {OUTPUT_ALPHA_DIR}")