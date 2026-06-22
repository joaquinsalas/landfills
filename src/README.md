# Directorio de pruebas y prototipos

Este directorio contiene los experimentos historicos y activos del proyecto de
segmentacion de rellenos sanitarios en imagenes satelitales.

La direccion cientifica recomendada es convertir estos prototipos en un
benchmark reproducible para comparar backbones/foundation models bajo el mismo
dataset, protocolo de entrenamiento y metricas.

## Estructura actual

```text
code/
├── u-net_dinov3_sat/     # prueba U-Net + DINOv3 ViT-L/16 satelital
└── u-net_vanilla/        # carpeta historica; actualmente no es vanilla real
```

## Rol de cada modelo

`u-net_dinov3_sat/` es la prueba en la que se estaba integrando U-Net con
DINOv3 ViT-L/16 como encoder.

`u-net_vanilla/` prueba con una u-net vanilla sin un backbone preentrenado especifico.

## Reglas para evolucionar este directorio

- Cada experimento debe documentar dataset, pesos usados, estrategia de
  entrenamiento, metricas y limitaciones.
- Los resultados de corridas no deberian mezclarse permanentemente con el
  codigo; a futuro deben vivir en `outputs/` en cada directorio.

## Prioridad recomendada

1. Auditar y documentar dataset.
2. Crear baseline U-Net real.
3. Consolidar loader, transforms, losses y metricas comunes.
