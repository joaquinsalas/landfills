import torch
import torch.nn as nn
import torch.optim as optim
import csv
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

# imports de los demas codigos
import config as ini
from Res_Unet34 import ResUNet34
from loader import prepare_data
from IoU import calculate_iou
from diceloss import DiceLoss
from plot import generate_metrics_plot

# sirve para que la memoria de la tarjeta de video no se fragmente y alcance para más
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_model():
    # Si no existe la carpeta donde se guardarán las cosas se crea.
    if not os.path.exists(ini.OUTPUT_DIR):
        os.makedirs(ini.OUTPUT_DIR)
        
    # Prepara herramientas para acelerar matematicas si se usa una tarjeta de video (GPU)
    scaler = torch.amp.GradScaler('cuda')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    # Trae la base de datos para que acomode los paquetes de fotos
    train_loader, val_loader = prepare_data()
    
    # 1. El Modelo (La U-Net con ResNet34 como cerebro)        
    model = ResUNet34().to(device) 

    # Píxel por píxel (BCE) y Formas (Dice)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    criterion_dice = DiceLoss()

    # El Optimizador para corregir a la ia
    # El lr=5e-5 es el ritmo al que aprende
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # 1. EL SCHEDULER Para bajar el ritmo de aprendizaje si se estanca el error de validación
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Crea el archivo Excel  
    log_path = os.path.join(ini.OUTPUT_DIR, "training_logs_resnet34.csv")
    with open(log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Escribe los títulos de las columnas
        writer.writerow(['epoch', 'train_loss', 'train_iou', 'val_loss', 'val_iou'])

# 2. Para guardar el mejor
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
                
                # 3. 20% es el BCE y 80% es el Dice
                loss = (0.20 * criterion_bce(outputs, masks)) + (0.80 * criterion_dice(outputs, masks))
            
            # Hace que los errores "fluyan hacia atrás" para corregir los cables cerebrales de la IA
            scaler.scale(loss).backward()
            scaler.step(optimizer) # Aplica la corrección aprendida
            scaler.update()

            # Anota los resultados de este de entrenamiento
            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks)

        # Validación
        model.eval() 
        val_loss, val_iou = 0.0, 0.0
        
        with torch.no_grad(): 
            for images, masks in val_loader: # Le pasa fotos de los resultados no visto
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                v_loss = (0.20 * criterion_bce(outputs, masks)) + (0.80 * criterion_dice(outputs, masks))
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
    # Avisa al Scheduler como va el error de validación
        scheduler.step()
        if metrics['val_iou'] > best_val_iou:
            best_val_iou = metrics['val_iou']
            torch.save(model.state_dict(), os.path.join(ini.OUTPUT_DIR, "resunet34_landfills_best.pth"))
           
    
    # Llama al codgio que dibuja las graficas de la resbaladilla y la rampa
    generate_metrics_plot(log_path)
    print(f"Modelo y logs guardados en {ini.OUTPUT_DIR}")

# Si se ejecuta este archivo, empieza todo el proceso
if __name__ == "__main__":
    train_model()