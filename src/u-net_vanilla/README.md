# U-Net vanilla

Este directorio está destinado al baseline U-Net del proyecto. Su función es
servir como punto de comparación principal frente a modelos con backbones
preentrenados o foundation models.

La idea de esta línea es entrenar una U-Net desde cero sobre el mismo dataset de
rellenos sanitarios, usando el mismo protocolo de evaluación que el resto de los
modelos.

## Objetivo experimental

Evaluar qué tan bien segmenta una arquitectura U-Net clásica las zonas de
relleno sanitario en imágenes satelitales.

Este resultado permite medir con claridad cuánto aportan modelos más complejos,
por ejemplo:

- U-Net + DINOv3 ViT-L/16.
- U-Net + DINOv3 ViT-7B/16.
- U-Net + otros backbones preentrenados.
- Representaciones geoespaciales externas.

## Estructura

```text
u-net_vanilla/
├── config/        # configuración del experimento
├── dataset/       # dataset PyTorch y transformaciones
├── evaluations/   # métricas y gráficas
├── models/        # arquitectura U-Net
└── training/      # entrenamiento, losses y loaders
```

## Rol dentro del benchmark

Este modelo debe funcionar como baseline controlado:

- sin pesos preentrenados
- sin backbones externos
- con encoder y decoder convolucionales
- con las mismas particiones `train/` y `val`
- con las mismas métricas usadas por los demás modelos

El resultado de esta línea debe responder:

> Qué desempeño obtiene una U-Net clásica antes de agregar backbones o modelos
> fundacionales?

## Outputs esperados

Cada corrida de este modelo debe guardar sus resultados en una carpeta propia,
por ejemplo:

```text
outputs/unet_vanilla/
├── checkpoints/
├── predictions/
│   ├── train/
│   └── val/
├── training_logs.csv
├── metrics_summary.png
├── metrics.json
└── config_snapshot.json
```

## Métricas mínimas

Para que una corrida pueda compararse con otros modelos debe reportar:

- IoU
- Dice/F1
- precision
- recall
- loss de validación
- número de parámetros
- tiempo aproximado de entrenamiento

## Criterios para considerar completo el modelo

- La arquitectura corresponde a una U-Net clásica.
- Entrena sobre el dataset común del proyecto.
- Usa los mismos splits que los demás modelos.
- Genera predicciones para validación.
- Guarda checkpoint, logs, métricas y configuración.
- Documenta cualquier limitación observada durante el entrenamiento o la
  inferencia.
