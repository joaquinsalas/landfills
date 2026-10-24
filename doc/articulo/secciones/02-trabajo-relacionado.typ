#import "../template.typ": *

= Trabajo relacionado <sec:related>

#guia[Objetivo: ubicar tu trabajo en la literatura y justificar la brecha que cubres. Apunta a 15–25 referencias revisadas por pares. Termina cada subsección con una frase que conecte con tu propuesta. Agrega cada fuente a references.bib y cítala con `@clave`.]

== Detección de rellenos sanitarios y residuos por percepción remota <sec:rel-residuos>

#todo[Revisión de la literatura: detección de vertederos y rellenos sanitarios con imágenes satelitales o aéreas, índices espectrales, métodos clásicos y redes convolucionales. Citar al menos 4–6 trabajos.]

== Segmentación semántica en imágenes satelitales <sec:rel-segmentacion>

La arquitectura U-Net @ronneberger2015unet, concebida originalmente para segmentación de imágenes biomédicas, se ha convertido en una referencia para segmentación densa gracias a sus conexiones de salto entre codificador y decodificador. Es habitual sustituir su codificador por redes preentrenadas como las redes residuales @he2016resnet o transformers de visión @dosovitskiy2021vit. #todo[Agregar aplicaciones de U-Net y variantes en percepción remota, con cita.]

== Modelos de fundación para observación de la Tierra <sec:rel-fundacion>

DINOv3 @simeoni2025dinov3 es una familia de modelos visuales entrenados mediante aprendizaje autosupervisado, de la cual existe una variante preentrenada con imágenes satelitales. AlphaEarth Foundations @brown2025alphaearth genera representaciones (embeddings) de la superficie terrestre a partir de múltiples fuentes de observación. #todo[Verificar y completar la descripción técnica de ambos modelos (datos de preentrenamiento, resolución, dimensión del embedding) contra los artículos originales.]

== Brecha de investigación <sec:rel-brecha>

#todo[Una frase que resuma lo que falta en la literatura y que este trabajo cubre: comparación sistemática de modelos de fundación para detección de rellenos sanitarios en México con pocos datos etiquetados.]
