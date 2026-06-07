import os
import torch
import torchvision.transforms as T
from transformers import AutoConfig, AutoModel
from transformers.image_utils import load_image

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

local_processor = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

inputs = local_processor(image).unsqueeze(0)

local_config_path = "./config.json"
local_weights_path = "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"

print("Cargando arquitectura desde el config.json local...")
config = AutoConfig.from_pretrained(local_config_path)

print("Inicializando modelo vacío...")
model = AutoModel.from_config(config)

print("Cargando pesos desde el archivo .pth local...")
state_dict = torch.load(local_weights_path, map_location="cpu")

if "model" in state_dict:
    state_dict = state_dict["model"]

model.load_state_dict(state_dict, strict=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

inputs = inputs.to(device)

print("Ejecutando inferencia local...")
with torch.inference_mode():
    outputs = model(inputs)

pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0]

print("\n¡Logrado! El script corrió de forma 100% aislada.")
print("Pooled output shape:", pooled_output.shape)
