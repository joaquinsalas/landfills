# U-Net + DINOv3 ViT-L/16 satelital

Este directorio contiene la prueba donde se estaba integrando una arquitectura
U-Net con DINOv3 ViT-L/16 como encoder/backbone.

Es una de las lineas principales para el articulo, porque representa un modelo
intermedio: mas viable que ViT-7B en costo computacional, pero mas potente que
un U-Net entrenado desde cero.

## Objetivo experimental

Evaluar el rendimiento de DINOv3 ViT-L/16 satelital como extractor de features
para segmentacion binaria de rellenos sanitarios.

Comparaciones deseadas:

- U-Net vanilla real.
- U-Net + ResNet/EfficientNet/Swin.
- U-Net + DINOv3 ViT-L/16 congelado.
- U-Net + DINOv3 ViT-L/16 fine-tuned parcial.
- DINOv3 ViT-7B.
- AlphaEarth.

## Estructura

```text
u-net_dinov3_sat/
├── config/        # configuracion del experimento
├── dataset/       # dataset PyTorch y transforms
├── evaluations/   # IoU y graficas
├── models/        # backbone, decoder y U-Net
├── training/      # loop de entrenamiento
├── test.py        # inferencia/visualizacion actual
└── environment.yml
```

## Estado actual

Esta carpeta fue saneada parcialmente para:

- quitar rutas absolutas antiguas
- usar `config.py` como centro de configuracion
- agregar `environment.yml`
- usar imports de paquete
- controlar inferencia desde `config.py`

Los cambios estan pendientes en Git y deben revisarse antes de consolidarse.

## Dataset esperado

Por defecto, despues del saneamiento parcial:

```text
data/squared/
├── train/
│   ├── landfills/
│   └── masks/
└── val/
    ├── landfills/
    └── masks/
```

Pero el dataset real del proyecto completo vive en:

```text
dataset/squared/
```

Antes de correr entrenamientos, hay que decidir si esta prueba se ajusta a la
raiz del proyecto o si se migra a una estructura comun.

## Flujo actual

Entrenamiento:

```bash
python training/trainer.py
```

Inferencia:

```bash
python test.py
```

Ambos dependen de `config/config.py`.

## Componentes importantes

- `models/model_unet.py`: reconstruye mapas espaciales desde tokens del ViT y
  los usa como skips para el decoder.
- `models/backbone.py`: carga DINOv3 via `torch.hub` o modelos `timm`.
- `training/trainer.py`: BCE + Dice, IoU y guardado de pesos.
- `test.py`: genera overlays con `supervision`.

## Riesgos conocidos

- El nombre `test.py` deberia cambiar a `predict.py` o `inference.py`.
- El modelo depende de hooks sobre `self.backbone.blocks`.
- Hay que validar que `BACKBONE_NAME` corresponda exactamente al modelo DINOv3
  satelital disponible.
- El entrenamiento todavia no guarda `best_iou`, `best_dice`, resumen JSON ni
  config usada por corrida.
- La evaluacion es basica; faltan Dice, precision, recall y analisis por imagen.

## Rol futuro

Esta prueba deberia convertirse en el experimento:

```text
configs/experiments/unet_dinov3_vitl16.py
```

La logica reutilizable deberia migrarse a:

```text
src/models/backbones/dinov3_vitl16.py
src/models/decoders/unet_decoder.py
src/training/
src/evaluation/
```
