#import "../template.typ": *

= Introducción <sec:intro>

#guia[Estructura en embudo, 4 a 6 párrafos (0.75–1.25 páginas): (1) contexto amplio, (2) problema concreto, (3) estado del arte y brecha, (4) propuesta y contribuciones, (5) organización del documento. Toda afirmación factual necesita cifra y cita. Escribir esta sección casi al final.]

// Párrafo 1: contexto
#todo[Contexto: gestión de residuos sólidos en México y su impacto ambiental (lixiviados, emisiones de metano, contaminación de acuíferos). Incluir una o dos cifras oficiales con su cita.]

// Párrafo 2: problema
#todo[Problema: los inventarios de sitios de disposición final están incompletos o desactualizados y existen sitios no regulados. Explicar por qué el monitoreo en campo no escala a nivel nacional.]

// Párrafo 3: oportunidad y brecha
#todo[Oportunidad: imágenes satelitales y segmentación semántica permiten un monitoreo a gran escala. Brecha: trabajos previos (citar) no cubren México, no comparan modelos de fundación o asumen grandes volúmenes de datos etiquetados.]

// Párrafo 4: propuesta
En este trabajo se aborda la detección de rellenos sanitarios en imágenes satelitales de México como un problema de segmentación semántica binaria, en un régimen de pocos datos etiquetados (menos de 100 imágenes con máscaras). Las contribuciones son las siguientes:

+ Se compara de forma controlada un conjunto de #n-modelos configuraciones de segmentación (@tab:modelos), que combinan codificadores preentrenados y arquitecturas U-Net, bajo el mismo conjunto de datos, particiones y protocolo de evaluación.
+ Se evalúa el aporte de las representaciones de modelos de fundación para observación de la Tierra (DINOv3 y AlphaEarth) frente a una línea base entrenada sin preentrenamiento.
+ #todo[Contribución adicional: conjunto de datos etiquetado, código abierto, mapa de predicción nacional, etc. Borrar si no aplica.]

// Párrafo 5: organización
El resto del documento se organiza como sigue. La @sec:related revisa el trabajo relacionado; la @sec:datos describe los datos; la @sec:metodos detalla la metodología; la @sec:resultados presenta los resultados, que se interpretan en la @sec:discusion; y la @sec:conclusiones resume las conclusiones y el trabajo futuro.
