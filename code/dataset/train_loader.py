from transform import get_transforms
from dataset import get_dataloader

import sys
sys.path.append('/home/emilio/Work/landfills/code/config/')
import config as ini

# obtiene las transformaciones
train_tf, val_tf, img_norm = get_transforms(ini.INPUT_SIZE)

# define rutas de entrada
TRAIN_IMG_PATH = f"{ini.DATA_DIR}/train/landfillls"
TRAIN_MASK_PATH = f"{ini.DATA_DIR}/train/masks"

# crea el data loader de entrenamiento
train_loader = get_dataloader(
    img_dir=TRAIN_IMG_PATH,
    mask_dir=TRAIN_MASK_PATH,
    batch_size=ini.BATCH_SIZE,
    transform=train_tf,
    img_norm=img_norm,
    num_workers=ini.NUM_WORKERS
)

# bloque de verificacion
images, masks = next(iter(train_loader))

print(f"--- Verificación de Dimensiones ---")
print(f"Dimensiones de Imágenes: {images.shape}")
print(f"Dimensiones de Máscaras: {masks.shape}")

print(f"\n--- Verificación de Tipos y Rangos ---")
print(f"Tipo de dato Imagen: {images.dtype} | Rango: [{images.min():.2f}, {images.max():.2f}]")
print(f"Tipo de dato Máscara: {masks.dtype} | Rango: [{masks.min():.2f}, {masks.max():.2f}]")

print(f"\n--- Análisis de la Máscara ---")
valores_unicos = torch.unique(masks)
print(f"Valores únicos encontrados en la máscara: {valores_unicos}")

if len(valores_unicos) > 2:
    print("Alerta: La máscara tiene más de 2 valores. Revisa la interpolación al redimensionar.")
