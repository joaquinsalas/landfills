import matplotlib
matplotlib.use('Agg')
import torch
import sys
import os

_f_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_f_dir) != "parts" and _f_dir != os.path.dirname(_f_dir):
    _f_dir = os.path.dirname(_f_dir)
PART_DIR = _f_dir

sys.path.append(PART_DIR)
sys.path.append(os.path.join(PART_DIR, "code", "config"))
sys.path.append(os.path.join(PART_DIR, "code", "models"))
sys.path.append(os.path.join(PART_DIR, "code", "dataset"))
sys.path.append(os.path.join(PART_DIR, "code", "evaluations"))
sys.path.append(os.path.join(PART_DIR, "code", "training"))

import config as ini
from Res_Unet34 import ResUNet34
from loader import prepare_data
from plot import visualizar_predicciones

def test_visual():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Procesando con: {device}")

    print("Cargando imágenes de prueba...")
    _, val_loader = prepare_data()

    model = ResUNet34(in_channels=64).to(device)

    ruta_pesos = os.path.join(ini.OUTPUT_DIR, "resunet34_landfills_best.pth")
    if os.path.exists(ruta_pesos):
        model.load_state_dict(torch.load(ruta_pesos, map_location=device))
        print("Pesos cargados correctamente.")
    else:
        print(f"No se encontró: {ruta_pesos}")
        return

    # Ruta de los .tif originales de val
    tif_dir = os.path.join(ini.DATA_DIR, "squared", "dataset", "squared", "val", "landfills")

    print("Generando visualización...")
    ruta_png = os.path.join(ini.OUTPUT_DIR, "predicciones_alphaearth.png")
    visualizar_predicciones(
        model, val_loader, device,
        num_imagenes=3,
        save_path=ruta_png,
        images_dir=tif_dir
    )

if __name__ == "__main__":
    test_visual()