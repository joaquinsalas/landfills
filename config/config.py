# Rutas 
DATA_DIR = r"C:\Users\rafae\OneDrive\Tareas\.vscode\proyectoML\parts\dataset" # Donde están tus fotos de vertederos
OUTPUT_DIR = r"C:\Users\rafae\OneDrive\Tareas\.vscode\proyectoML\parts\output" # Donde se guardarán las gráficas y el modelo
# Configuración de las imágenes
INPUT_SIZE = 256 
# Configuración del entrenamiento
SPLITS = ['train', 'val'] # una carpeta para entrenar (train) y otra para la validación (val)
BATCH_SIZE = 2 # La computadora va a procesar las fotos de 2 en 2 para no saturar la RAM
NUM_WORKERS = 1 # Procesos en segundo plano para cargar datos 
EPOCHS = 20 # Cuántas veces va a repasar la IA toda la base de datos 