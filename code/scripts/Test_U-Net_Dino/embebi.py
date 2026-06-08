from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch
import os

# Carpeta donde están tus imágenes
dataset_path = "C:/proyect_gaby/landfills/dataset/squared/train/landfills" 

# Cargar todas las imágenes de la carpeta
imagenes = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png', '.tif'))]

modelo = "facebook/dinov2-large"

processor = AutoImageProcessor.from_pretrained(modelo)
model = AutoModel.from_pretrained(modelo).eval()

# Procesar cada imagen
for nombre_imagen in imagenes:
    ruta_completa = os.path.join(dataset_path, nombre_imagen)
    
    image = Image.open(ruta_completa).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.pooler_output
    
    print(f"Imagen: {nombre_imagen}")
    print(f"Embedding shape: {embedding.shape}")
    print(embedding)
    print("---")