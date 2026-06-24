# config/config.py
# ─────────────────────────────────────────────────────────────────────
# Configuracion central del experimento.
# Todos los demas modulos importan desde aqui.
# ─────────────────────────────────────────────────────────────────────
import torch
from pathlib import Path

# ─── MODELO ───────────────────────────────────────────────────────────
MODELO_ENCODER = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # Encoder principal
MODELO_FALLBACK = "facebook/dinov2-large"                     # Fallback si no hay acceso
TOKEN_HF       = "hf"                          # Token de HuggingFace

# ─── ENTRENAMIENTO ────────────────────────────────────────────────────
IMG_SIZE    = 224    # Tamaño de imagen de entrada
BATCH_SIZE  = 4      # Imagenes por batch
EPOCHS      = 60     # Epocas totales
LR_DECODER  = 1e-4   # Learning rate del decoder
LR_ENCODER  = 1e-5   # Learning rate del encoder (mas bajo para fine-tuning)
WEIGHT_DECAY = 1e-4  # Regularizacion L2
UNFREEZE_EPOCH = 10  # Epoca en la que se descongela el encoder
UNFREEZE_LAYERS = 4  # Numero de capas del encoder a descongelar

# ─── RUTAS ────────────────────────────────────────────────────────────
# RAIZ_PROYECTO = carpeta que contiene config/, models/, outputs/, etc.
# Se calcula automaticamente, asi que outputs/ siempre vive dentro del
# proyecto sin importar desde donde se ejecute el script.
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

# Imagenes y mascaras de train/val: carpeta externa al proyecto (ruta fija).
# En Windows, usa r"..." para que las barras invertidas no se interpreten
# como caracteres de escape.
TRAIN_IMG  = Path(r"C:\proyect_gaby\landfills\dataset\train\landfills")
TRAIN_MASK = Path(r"C:\proyect_gaby\landfills\dataset\train\masks")
VAL_IMG    = Path(r"C:\proyect_gaby\landfills\dataset\val\landfills")
VAL_MASK   = Path(r"C:\proyect_gaby\landfills\dataset\val\masks")

# Salidas (modelo, embeddings, graficas): dentro del proyecto, en outputs/
OUTPUTS_DIR         = RAIZ_PROYECTO / "outputs"
MODELO_PTH          = OUTPUTS_DIR / "modelo_dualencoder_dino_resnet.pth"
EMBEDDINGS_NPY      = OUTPUTS_DIR / "embeddings.npy"
EMBEDDINGS_NAMES    = OUTPUTS_DIR / "embeddings_nombres.txt"
SALIDA_METRICAS_PNG = OUTPUTS_DIR / "metricas_dualencoder.png"

# ─── DEVICE ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── NORMALIZACION ImageNet (usada por DINOv2/v3) ────────────────────
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# ─── EXTENSIONES VALIDAS ──────────────────────────────────────────────
EXTENSIONES = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
