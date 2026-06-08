from PIL import Image  # Para abrir y manipular imagenes
import torch  # Motor de deep learning
import torch.nn as nn  # Modulo de redes neuronales
import torchvision.transforms as T  # Para transformar imagenes a tensores
from torch.utils.data import Dataset, DataLoader  # Para cargar el dataset en batches
from modelo import UNetDINO  # Importa solo la clase del modelo
import os  # Para navegar carpetas y archivos

# ─── CONFIG ───────────────────────────────────────────────────────────
IMG_SIZE   = 224  # Tamaño al que se redimensionan las imagenes
BATCH_SIZE = 4  # Cuantas imagenes se procesan a la vez
EPOCHS     = 20  # Cuantas veces recorre todo el dataset
LR         = 1e-4  # Tasa de aprendizaje
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU o CPU

TRAIN_IMG  = r"C:\proyect_gaby\landfills\dataset\squared\train\landfills"
TRAIN_MASK = r"C:\proyect_gaby\landfills\dataset\squared\train\masks"

print(f"Usando: {DEVICE}")

# ─── DATASET ──────────────────────────────────────────────────────────
class LandfillDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=IMG_SIZE):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.size     = size

        # Lista todos los archivos de imagen en la carpeta
        self.imagenes = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))
        ])

        # Transformaciones para las imagenes
        self.img_transform = T.Compose([
            T.Resize((size, size)),  # Redimensiona a 224x224
            T.ToTensor(),  # Convierte a tensor
            T.Normalize(mean=[0.485, 0.456, 0.406],  # Normaliza con valores de ImageNet
                        std =[0.229, 0.224, 0.225])
        ])

        # Transformaciones para las mascaras
        self.mask_transform = T.Compose([
            T.Resize((size, size)),  # Redimensiona a 224x224
            T.ToTensor()  # Convierte a tensor
        ])

    def __len__(self):
        return len(self.imagenes)  # Retorna el numero total de imagenes

    def __getitem__(self, idx):
        nombre = os.path.splitext(self.imagenes[idx])[0]  # Nombre sin extension

        # Cargar imagen
        img_path = os.path.join(self.img_dir, self.imagenes[idx])
        image = Image.open(img_path).convert("RGB")  # Abre en RGB

        # Buscar mascara con cualquier extension
        mask = None
        for ext in ('.tif', '.tiff', '.png', '.jpg'):
            mask_path = os.path.join(self.mask_dir, nombre + ext)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")  # L = escala de grises
                break

        if mask is None:
            raise FileNotFoundError(f"No se encontro mascara para: {nombre}")

        image = self.img_transform(image)
        mask  = self.mask_transform(mask)
        mask  = (mask > 0.5).float()  # Binariza la mascara: 0 o 1

        return image, mask

# ─── ENTRENAMIENTO ────────────────────────────────────────────────────
dataset    = LandfillDataset(TRAIN_IMG, TRAIN_MASK)  # Crea el dataset
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # Carga en batches

print(f"Imagenes encontradas: {len(dataset)}")

model     = UNetDINO().to(DEVICE)  # Crea el modelo y lo manda a GPU o CPU
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Optimizador Adam
criterion = nn.BCELoss()  # Funcion de perdida para segmentacion binaria

for epoch in range(EPOCHS):
    model.train()  # Modo entrenamiento
    total_loss = 0

    for images, masks in dataloader:
        images = images.to(DEVICE)  # Manda imagenes a GPU o CPU
        masks  = masks.to(DEVICE)   # Manda mascaras a GPU o CPU

        optimizer.zero_grad()       # Limpia gradientes del paso anterior
        preds = model(images)       # Predice la mascara
        loss  = criterion(preds, masks)  # Calcula el error
        loss.backward()             # Calcula gradientes
        optimizer.step()            # Actualiza los pesos

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# ─── GUARDAR MODELO ───────────────────────────────────────────────────
torch.save(model.state_dict(), r"C:\proyect_gaby\landfills\modelo_unet_dino.pth")  # Guarda los pesos
print("Modelo guardado.")