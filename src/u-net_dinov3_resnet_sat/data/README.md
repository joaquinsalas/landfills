# Carpeta data/

Coloca aqui tus imagenes y mascaras de entrenamiento/validacion:

```
data/
├── train/
│   ├── images/   <- imagenes de entrenamiento (.tif, .tiff, .jpg, .jpeg, .png)
│   └── masks/    <- mascaras de entrenamiento (mismo nombre que su imagen)
└── val/
    ├── images/   <- imagenes de validacion
    └── masks/    <- mascaras de validacion
```

Cada mascara debe tener el MISMO NOMBRE BASE que su imagen correspondiente
(la extension puede variar, dataset.py la busca automaticamente).

Ejemplo:
  train/images/relleno_001.jpg
  train/masks/relleno_001.png
