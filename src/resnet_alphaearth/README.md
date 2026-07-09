# Segmentación de Rellenos Sanitarios con AlphaEarth + ResUNet34

## Objetivo experimental

Este proyecto evalua si los **embeddings geoespaciales de AlphaEarth** (modelo fundacional de Google DeepMind) mejoran la segmentación semántica de rellenos sanitarios frente a un enfoque tradicional con imagenes RGB.

Se entrenan y comparan dos variantes del mismo modelo:

- **Modelo RGB**: entrada de 3 canales (imagen satelital visual estándar)
- **Modelo AlphaEarth**: entrada de 64 canales (tensor de embeddings `[64, 512, 512]` extraído del dataset `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`)

El objetivo final es determinar si un **ensemble** de ambos modelos supera el desempeño individual de cada uno, dado un dataset reducido de 47 polígonos.

-  Objetivo experimental definido


##  Arquitectura y dependencias principales

| Componente | Detalle |
|---|---|
| Framework | PyTorch |
| Backbone | ResNet-34 preentrenado (ImageNet) |
| Arquitectura | U-Net con skip connections (ResUNet34) |
| Fuente de embeddings | Google Earth Engine — `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (AlphaEarth) |
| Pérdida | `BCEWithLogitsLoss(pos_weight=3.0)` + `DiceLoss` (ponderación 20%/80%) |
| Optimizador | Adam, `lr=5e-5` |
| Scheduler | `CosineAnnealingLR` |
| Precisión mixta | `torch.amp.GradScaler` (solo en entrenamiento, CUDA) |
| Métrica principal | IoU (Intersection over Union) |
| Librerías clave | `torch`, `torchvision`, `ee` (Earth Engine API), `rasterio`, `numpy`, `pandas`, `matplotlib`, `PIL` |

- Arquitectura y dependencias documentadas

##  Estructura interna de la carpeta

code/
│   ├── config/
│   │   └── config.py          # Rutas dinámicas 
│   │
│   ├── dataset/
│   │   ├── dataset.py         # Clase LandfillDataset — lee tensores .pt + mascaras .tif ahora procesadas para que ne los tensores no sea rgb de 3 dimensiones sino que sean 64
│   │   └── transform.py       # Augmentaciones (None para AlphaEarth, flips/rotación para RGB)
│   │
│   ├── models/
│   │   └── Res_Unet34.py      # Arquitectura ResUNet34 (encoder ResNet-34 + decoder U-Net) se usa esta estructura ya que no es un dataset muy grande por el momento
│   │
│   ├── training/
│   │   ├── loader.py          # prepare_data() — arma train_loader y val_loader
│   │   ├── diceloss.py        # Pérdida Dice (cálculo por imagen, no por batch aplanado)
│   │   └── trainer.py         # Loop principal de entrenamiento y validación
│   │
│   └── evaluations/
│       ├── extract_alphaearth.py  # Extrae embeddings AlphaEarth desde Earth Engine → tensores .pt
│       ├── IoU.py                 # Cálculo de la métrica IoU
│       ├── plot.py                # Gráficas de métricas + comparativas visuales de predicciones
│       └── View_r.py              # Script de inferencia visual sobre el set de validación
│

Estructura de carpetas documentada

## Pasos de réplica

**1. Extraer embeddings de AlphaEarth** (requiere autenticación en Google Earth Engine):
```bash
cd parts/code/evaluations
python extract_alphaearth.py
```

**2. Verificar integridad de los tensores generados:**
```bash
python test_tensors.py
```
**O simplemente con los .tiff ya hecho y adaptando el training para estos nuevos valores de los tensores**

**3. Probar el pipeline de datos antes de entrenar:**
```bash
cd ../training
python loader.py
```

**4. Entrenar el modelo:**
```bash
python trainer.py
```

**5. Generar gráficas de métricas:**
```bash
cd ../evaluations
python plot.py
```

**6. Visualizar predicciones sobre el set de validación:**
```bash
python View_r.py
```
- Pasos de réplica documentados

---

## Outputs esperados

| Archivo | Ubicación | Contenido |
|---|---|---|
| `resunet34_landfills_best.pth` | `parts/output/` | Checkpoint con mejor Val IoU (modelo con AlphaEarth) |
| `training_logs_resnet34.csv` | `parts/output/` | epoch, train_loss, train_iou, val_loss, val_iou |
| `metrics_summary.png` | `parts/output/` | Gráficas de Loss y IoU por época |
| `predicciones_alphaearth.png` | `parts/output/` | Comparativa visual: imagen real / máscara real / predicción |
| `patches/*.pt` | `parts/output/patches/` | Tensores de embeddings AlphaEarth `[64,512,512]` |

## Métricas reportadas

- **IoU (Intersection over Union)** — métrica principal, calculada por batch con threshold configurable (0.5–0.6)
- **BCE Loss** (Binary Cross-Entropy con `pos_weight` para compensar desbalance de clases)
- **Dice Loss** (calculada por imagen individual dentro del batch)

**Resultados actuales (validación, 9 imágenes):**

| Modelo | Val IoU máximo | Épocas |
|---|---|---|
| RGB (3 canales) | ~0.65 | 80 |
| AlphaEarth (64 canales) | ~0.65 | 150 |


## Limitaciones conocidas

- **Dataset muy reducido**: solo 47 poligonos (38 train / 9 val) alto riesgo de overfitting y metricas de validacion ruidosas.
- **Embeddings de AlphaEarth no admiten augmentacion geometrica estandar**: al ser representaciones abstractas preprocesadas, rotaciones/flips con `torchvision.transforms` pueden distorsionar la informacion codificada; actualmente se entrena sin augmentacion para esta rama.
- **Primera capa del encoder sin pesos preentrenados**: al pasar de 3 a 64 canales de entrada, la capa `conv1` se reinicializa aleatoriamente y pierde el beneficio del preentrenamiento en ImageNet para esa capa especifica.
- **Dependencia de cuota de Google Earth Engine**: la extracion de embeddings requiere autenticacion interactiva y esta sujeta a límites de la API (`getDownloadURL`).
- **Bordes ruidosos en polígonos grandes**: el modelo AlphaEarth mostro fragmentacion en la segmentación de vertederos de mayor tamaño, pendiente de investigar (posible relación con el padding aplicado a tensores menores a 512×512).
- **Ausencia de Dropout y Early Stopping** en la version actual del entrenamiento — mejoras identificadas pero aún no implementadas.


## Estado actual del experimento

**En desarrollo.**

-  Pipeline de extraccion de embeddings AlphaEarth funcional y validado (47/47 tensores correctos)
-  Pipeline de datos (Dataset/DataLoader) funcional para ambas variantes (RGB y AlphaEarth)
-  Modelo ResUNet34 adaptado exitosamente a entrada de 64 canales
-  Entrenamiento estable tras corregir DiceLoss (calculo por imagen), scheduler y `pos_weight`
-  Ambos modelos (RGB y AlphaEarth) alcanzan Val IoU ~0.65 de forma independiente
-  Pendiente: Dropout, Early Stopping, aumento de epocas, augmentacion específica para rama RGB
-  Pendiente: Ensemble de ambos modelos (RGB + AlphaEarth +)
-  Pendiente: Benchmark final y comparacion cuantitativa formal
-  Pendiente: Mas pruebas de para una mejor optimizacion
-  Estado actual del experimento documentado

##Autor

**Alvarado Ibañez Rafael**