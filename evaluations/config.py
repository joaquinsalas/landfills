import os
# rutas dinamicas 
_f_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_f_dir) != "parts" and _f_dir != os.path.dirname(_f_dir):
    _f_dir = os.path.dirname(_f_dir)
PART_DIR = _f_dir
DATA_DIR = os.path.join(PART_DIR, "dataset") # dirección fotos de vertederos
OUTPUT_DIR = os.path.join(PART_DIR, "output") # Donde se guardarán las gráficas y el modelo
# tamaño de dimension de las imagenes
INPUT_SIZE = 512 
# Configuración del entrenamiento
SPLITS = ['train', 'val'] # una carpeta para entrenar (train) y otra para la validación (val)
BATCH_SIZE = 4 # cantidad de procesos en segundo plano
NUM_WORKERS = 1 # Procesos en segundo plano  
EPOCHS = 150 # epocas (repasos)