import torch
import torch.nn as nn
import torch.optim as optim
import csv
import sys
import os

# Esto le dice a Python en qué carpetas secretas de tu computadora debe buscar 
# las otras piezas de codigo que armaste (el modelo, las configuraciones, etc.)
sys.path.append(r'C:\Users\rafae\OneDrive\Tareas\.vscode\proyectoML\parts\code\config')
sys.path.append(r'C:\Users\rafae\OneDrive\Tareas\.vscode\proyectoML\parts\code\models')
sys.path.append(r'C:\Users\rafae\OneDrive\Tareas\.vscode\proyectoML\parts\code\dataset')
sys.path.append(r'C:\Users\rafae\OneDrive\Tareas\.vscode\proyectoML\parts\code\evaluations')

# imports de los demas codigos
import config as ini
from Res_Unet18 import ResUNet18
from loader import prepare_data
from IoU import calculate_iou
from diceloss import DiceLoss
from plot import generate_metrics_plot

# sirve para que la memoria de la tarjeta de video no se fragmente y alcance para más
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_model():
    # Si no existe la carpeta donde se guardarán las cosas, la crea.
    if not os.path.exists(ini.OUTPUT_DIR):
        os.makedirs(ini.OUTPUT_DIR)
        
    # Prepara herramientas para acelerar matematicas si se usa una tarjeta de video (GPU)
    scaler = torch.amp.GradScaler('cuda')
    
    # esto checa si tienes tarjeta NVIDIA ("cuda") o si usará el procesador normal ("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    # Trae la base de datos para que acomode los paquetes de fotos
    train_loader, val_loader = prepare_data()
    
    # Despierta a la IA  y la pone en la zona de trabajo (CPU)
    model = ResUNet18().to(device) # resnet18 es la versión mejorada de la U-Net, con un "cerebro" mas inteligente-

    # Píxel por píxel (BCE) y Formas (Dice)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    # El Optimizador para corregir a la ia
    # El lr=1e-4 es el ritmo al que aprende
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 1. EL SCHEDULER (Acelerador inteligente)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Crea el archivo Excel (.csv) 
    log_path = os.path.join(ini.OUTPUT_DIR, "training_logs.csv")
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Escribe los títulos de las columnas
        writer.writerow(['epoch', 'train_loss', 'train_iou', 'val_loss', 'val_iou'])

# 2. EL RASTREADOR DE RÉCORDS (Para guardar el mejor)
    best_val_iou = 0.0

    for epoch in range(ini.EPOCHS): # Repite esto por cada "época" que se configure
        
        #entranamiento
        model.train() # Pone a la IA a aprender
        train_loss, train_iou = 0.0, 0.0
        
        for images, masks in train_loader: # El bibliotecario le pasa paquetes de fotos
            images, masks = images.to(device), masks.to(device) # Las pone en la mesa de trabajo
            
            optimizer.zero_grad() # Borra lo que aprendio en la foto anterior para no confundirse
            
            with torch.amp.autocast('cuda'):
                outputs = model(images) # La IA dibuja lo que cree que es basura
                
                # 3. LOS NUEVOS PESOS (20% y 80%) en Entrenamiento
                loss = (0.2 * criterion_bce(outputs, masks)) + (0.8 * criterion_dice(outputs, masks))
                # Le suma los criterios de los píxeles (BCE) más los criterios de las formas (Dice)
                #loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
            
            # Hace que los errores "fluyan hacia atrás" para corregir los cables cerebrales de la IA
            scaler.scale(loss).backward()
            scaler.step(optimizer) # Aplica la corrección aprendida
            scaler.update()

            # Anota los resultados de este de entrenamiento
            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks)

        # Validación
        model.eval() # Pone a la IA a validar ya no puede corregir sus cables, solo responder
        val_loss, val_iou = 0.0, 0.0
        
        with torch.no_grad(): # Le quita el borrador ya no puede aprender aquí
            for images, masks in val_loader: # Le pasa fotos de los resultados no visto
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    # 3. LOS NUEVOS PESOS (20% y 80%) en Validación
                    v_loss = (0.2 * criterion_bce(outputs, masks)) + (0.8 * criterion_dice(outputs, masks))
                
                # Anota los resultados
                val_loss += v_loss.item()
                val_iou += calculate_iou(outputs, masks)

        # Saca los valores promedios de las epocas
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_iou': train_iou / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'val_iou': val_iou / len(val_loader)
        }

        # Imprime en la pantalla cómo le fue 
        print(f"Epoch [{metrics['epoch']}/{ini.EPOCHS}] "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"Val IoU: {metrics['val_iou']:.4f}")

        # Escribe los resultados en el Excel para no perderlos
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics['epoch'], 
                metrics['train_loss'], 
                metrics['train_iou'], 
                metrics['val_loss'], 
                metrics['val_iou']
            ])
    # Avisa al Scheduler cómo va el error de validación
        scheduler.step(metrics['val_loss'])
        if metrics['val_iou'] > best_val_iou:
            best_val_iou = metrics['val_iou']
    # Al terminar todas las épocas, guarda lo que aprendió la IA en un archivo .pth
            torch.save(model.state_dict(), f"{ini.OUTPUT_DIR}/resunet_landfills_best.pth")
    
    # Llama al codgio que dibuja las graficas de la resbaladilla y la rampa
    generate_metrics_plot(log_path)
    print(f"Modelo y logs guardados en {ini.OUTPUT_DIR}")

# Si se ejecuta este archivo, empieza todo el proceso
if __name__ == "__main__":
    train_model()