import os
import torch
import torchvision.transforms as T
from transformers import AutoConfig, AutoModel
from transformers.image_utils import load_image

# Apagamos por completo cualquier intento de conexión externa
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 1. Cargar la imagen (Si estás totalmente aislado de red, cambia esta URL por la ruta a una foto local)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

# 2. Reemplazo manual de AutoImageProcessor usando torchvision.transforms
# DINOv3 ViT-7B usa parches de 14x14 e imágenes escaladas y normalizadas con ImageNet
local_processor = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Procesamos y añadimos la dimensión de batch -> [1, 3, 224, 224]
inputs = local_processor(image).unsqueeze(0)

# 3. Definir las rutas locales de tus archivos en la carpeta actual
local_config_path = "./config.json"
local_weights_path = "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"

print("Cargando arquitectura desde el config.json local...")
config = AutoConfig.from_pretrained(local_config_path)

print("Inicializando modelo vacío...")
model = AutoModel.from_config(config)

print("Cargando pesos desde el archivo .pth local...")
state_dict = torch.load(local_weights_path, map_location="cpu")

# Meta suele guardar el state_dict real dentro de una llave llamada 'model' en el .pth
if "model" in state_dict:
    state_dict = state_dict["model"]

# Inyectamos los pesos en la estructura de Transformers
# strict=False evita crasheos si hay pequeñas diferencias de nomenclatura en las capas de Meta vs HF
model.load_state_dict(state_dict, strict=False)

# Enviamos a GPU si está disponible (Ojo: ¡Recuerda el consumo de VRAM de este modelo de 7B!)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Mover la imagen procesada al mismo dispositivo
inputs = inputs.to(device)

print("Ejecutando inferencia local...")
with torch.inference_mode():
    outputs = model(inputs)

# Extraemos el output
pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0]

print("\n¡Logrado! El script corrió de forma 100% aislada.")
print("Pooled output shape:", pooled_output.shape)
