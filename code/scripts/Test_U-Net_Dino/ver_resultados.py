from PIL import Image  # Para abrir y manipular imagenes
import torch  # Motor de deep learning
import torchvision.transforms as T  # Para transformar imagenes a tensores
import matplotlib.pyplot as plt  # Para mostrar y guardar graficas
import os  # Para navegar carpetas y archivos
from modelo import UNetDINO  # Importa solo la clase del modelo, Importa el modelo que entrenamos

# Define si usa GPU o CPU
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224  # Tamaño al que se redimensionan las imagenes

# Rutas de las imagenes y el modelo guardado
TRAIN_IMG  = r"C:\proyect_gaby\landfills\dataset\squared\train\landfills"
MODELO_PTH = r"C:\proyect_gaby\landfills\modelo_unet_dino.pth"

# Carga el modelo y sus pesos guardados despues del entrenamiento
model = UNetDINO().to(DEVICE)
model.load_state_dict(torch.load(MODELO_PTH, map_location=DEVICE, weights_only=True))  # Carga los pesos del archivo .pth
model.eval()  # Modo evaluacion: no actualiza pesos ni guarda gradientes

# Define las transformaciones que se aplican a cada imagen antes de entrar al modelo
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),  # Redimensiona la imagen a 224x224
    T.ToTensor(),  # Convierte la imagen a tensor de numeros entre 0 y 1
    T.Normalize(mean=[0.485, 0.456, 0.406],  # Normaliza los colores con los valores
                std =[0.229, 0.224, 0.225])   # estandar de ImageNet que uso DINOv2
])

# Toma los primeros 3 archivos de la carpeta de imagenes para visualizar
imagenes = sorted(os.listdir(TRAIN_IMG))[:3]

# Crea una figura con 3 filas y 2 columnas (imagen original | mascara predicha)
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

# Recorre cada imagen seleccionada
for i, nombre in enumerate(imagenes):
    img_path = os.path.join(TRAIN_IMG, nombre)  # Ruta completa de la imagen
    image    = Image.open(img_path).convert("RGB")  # Abre la imagen en RGB
    input_t  = transform(image).unsqueeze(0).to(DEVICE)  # Aplica transformaciones y agrega dimension de batch

    # Pasa la imagen por el modelo sin calcular gradientes (mas rapido y ahorra memoria)
    with torch.no_grad():
        pred = model(input_t)  # El modelo predice la mascara

    # Convierte la mascara predicha a numpy para poder graficarla
    pred_mask = pred.squeeze().cpu().numpy()  # squeeze quita dimensiones extra, cpu la mueve de GPU a RAM

    # Columna izquierda: imagen original
    axes[i, 0].imshow(image.resize((IMG_SIZE, IMG_SIZE)))
    axes[i, 0].set_title(f"Imagen: {nombre}")
    axes[i, 0].axis("off")  # Oculta los ejes

    # Columna derecha: mascara predicha en escala de calor (blanco=relleno, negro=no relleno)
    axes[i, 1].imshow(pred_mask, cmap="hot")
    axes[i, 1].set_title("Mascara predicha")
    axes[i, 1].axis("off")  # Oculta los ejes

plt.tight_layout()  # Ajusta el espacio entre graficas para que no se encimen
plt.savefig(r"C:\proyect_gaby\landfills\resultados.png", dpi=150, bbox_inches='tight')  # Guarda la imagen en disco
plt.close()  # Cierra la figura para liberar memoria
print("Imagen guardada en resultados.png")
print("Terminado")