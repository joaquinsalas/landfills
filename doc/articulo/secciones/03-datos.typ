#import "../template.typ": *

= Datos <sec:datos>

#guia[Esta sección debe permitir que otra persona reconstruya tu conjunto de datos. Reporta fuentes, resolución, fechas, criterio de etiquetado, tamaños exactos y particiones. Si no cabe, mueve detalles al apéndice.]

== Área de estudio y fuentes de imágenes <sec:datos-fuentes>

#todo[Cobertura geográfica (estados o regiones), fuente de las imágenes (proveedor, sensor), resolución espacial, fechas de captura y tamaño de parche en píxeles.]

#figura(
  fig-pendiente(alto: 5cm, texto: "Mapa de México con la ubicación de las muestras etiquetadas"),
  caption: [Distribución geográfica de las muestras. #todo[Describir qué muestran los marcadores.]],
) <fig:mapa>

== Etiquetado <sec:datos-etiquetas>

El conjunto etiquetado consta de menos de 100 imágenes con máscaras binarias (relleno sanitario y fondo). #todo[Cifra exacta de imágenes, proporción de píxeles positivos, criterio de etiquetado, herramienta utilizada, quién etiquetó y si hubo revisión cruzada.]

== Representaciones de entrada <sec:datos-entrada>

Se consideran dos tipos de entrada:

- *Embeddings de 3 dimensiones (RGB).* #todo[Cómo se obtienen y cómo se alinean con las máscaras.]
- *Embeddings de 64 dimensiones de AlphaEarth.* #todo[Año(s) del embedding, resolución, remuestreo y alineación con las máscaras.]

== Particiones y aumento de datos <sec:datos-particiones>

#todo[Estrategia de particiones entrenamiento, validación y prueba. Dado el tamaño reducido del conjunto, describir la validación cruzada y cómo se evita la fuga espacial (parches vecinos en particiones distintas). Aumentos aplicados.]

#tabla(
  ([Partición], [Imágenes], [Píxeles positivos (%)]),
  (
    ([Entrenamiento], todo[N], todo[%]),
    ([Validación], todo[N], todo[%]),
    ([Prueba], todo[N], todo[%]),
  ),
  caption: [Composición de las particiones del conjunto de datos.],
  columnas: (auto, 1fr, 1fr),
  alineacion: (left, center, center),
) <tab:particiones>
