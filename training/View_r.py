import torch
import sys
import os

# 1. rutas
sys.path.append(r'C:\Users\rafae\OneDrive\Tareas\.vscode\proyectoML\parts\code\config')
sys.path.append(r'C:\Users\rafae\OneDrive\Tareas\.vscode\proyectoML\parts\code\models')
sys.path.append(r'C:\Users\rafae\OneDrive\Tareas\.vscode\proyectoML\parts\code\dataset')
sys.path.append(r'C:\Users\rafae\OneDrive\Tareas\.vscode\proyectoML\parts\code\evaluations')

import config as ini
from Res_Unet18 import ResUNet18
from loader import prepare_data
from plot import visualizar_predicciones

def test_visual():
    # Detecta la tarjeta gráfica o el procesador normal
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Procesando con: {device}")

    # 2. Carga los datos a
    print("Cargando imágenes de prueba")
    _, val_loader = prepare_data()

    # 3. resnet18
    model = ResUNet18().to(device) # resnet18 es la versión mejorada de la U-Net, con un "cerebro" mas inteligente-

    # 4. pasar el conocimiento (modelo ya entrenado)
    ruta_pesos = f"{ini.OUTPUT_DIR}/resunet_landfills_best.pth" 
    
    if os.path.exists(ruta_pesos):
        # map_location=device asegura que no haya crasheos si pasas de GPU a CPU
        model.load_state_dict(torch.load(ruta_pesos, map_location=device))
        print("Pesos cargados.")
    else:
        print("No se encontro el archivo .pth guardado.")
    # 5. Llamar a la función de la gráfica
    print("Abriendo visualizador...")
    visualizar_predicciones(model, val_loader, device, num_imagenes=3)

if __name__ == "__main__":
    test_visual()