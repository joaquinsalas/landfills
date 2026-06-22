# Landfills

Segmentación binaria de rellenos sanitarios en imágenes satelitales. El
objetivo del proyecto es construir un benchmark reproducible para comparar
modelos de segmentación sobre el mismo dataset, protocolo de entrenamiento y
métricas.

La documentación técnica detallada vive de forma modular dentro de `src/`, con un README por
modelo o línea experimental.

## Estructura del proyecto

```text
.
├── dataset/
│   ├── train/
│   │   ├── landfills/      # imágenes de entrenamiento
│   │   └── masks/          # máscaras de entrenamiento
│   └── val/
│       ├── landfills/      # imágenes de validación
│       └── masks/          # máscaras de validación
├── metadata/
│   └── inegi/              # archivos fuente y tabulares de referencia
├── models/                 # pesos locales o externos usados por experimentos
└── src/                    # modelos, prototipos y líneas experimentales
    ├── u-net_vanilla/      # experimento con u-net vanilla
    └── u-net_dinov3_sat/   # experimento con u-net + dinov3 vitl16 SAT
```

El dataset actual contiene pares imagen/máscara. Cada imagen en `landfills/`
debe tener una máscara con el mismo nombre en `masks/`, por ejemplo:

```text
dataset/train/landfills/poligono_0.tif
dataset/train/masks/poligono_0.tif
```

## Documentación por modelo

Cada carpeta dentro de `src/` debe tener su propio README con:

- objetivo experimental del modelo
- arquitectura y dependencias principales
- estructura interna de la carpeta
- pasos de réplica
- outputs esperados
- métricas reportadas
- limitaciones conocidas
- estado actual del experimento
- autor

Tambien cada modelo debe tener su LICENCE dando credito a las herramientas usadas.

## Organización de outputs

Los outputs no deben mezclarse entre modelos. Cada experimento debe documentar
en su propio README dónde guarda resultados y debe separar sus corridas por
modelo, configuración o fecha.

```text
outputs/
└── version/
    ├── checkpoints/
    │   ├── best_iou.pth
    │   └── best_dice.pth
    │
    ├── logs/
    │   └── metrics.csv
    │
    ├── plots/
    │   └── training_curves.png
    │
    └── predictions/
```

Antes de dar por terminado un modelo, su README debe dejar claro qué outputs
produce y cuáles archivos son necesarios para reproducir o auditar la corrida.

## Cómo contribuir

- Mantener la documentación modular: la raíz explica el proyecto; cada modelo
  explica su réplica y resultados.
- Crear o actualizar el README del modelo cuando se agregue una carpeta nueva
  en `src/`.
- Usar el mismo dataset, splits y métricas cuando se comparen modelos.
- Conservar nombres idénticos entre imágenes y máscaras.
- No mezclar outputs de modelos distintos en la misma carpeta.
- Guardar configuración, métricas y checkpoints suficientes para reproducir una
  corrida.
- Documentar limitaciones conocidas en el README del modelo, no solo en código.
- No tratar una línea experimental como baseline hasta que su README y su código
  describan exactamente la misma arquitectura.
- Separar outputs por modelo y por corrida.
- Mantener actualizados los README modulares conforme cambien los experimentos.
