#import "../template.typ": *

= Resultados <sec:resultados>

#guia[Solo hechos: cifras, tablas y figuras, sin interpretar (eso va en la discusión). Cada tabla y figura debe entenderse sola gracias a su pie. La tabla de resultados se genera desde data/resultados.csv; las frases con cifras se calculan solas desde ese mismo archivo.]

== Comparación cuantitativa <sec:res-cuantitativa>

La @tab:resultados resume el desempeño de las #n-modelos configuraciones en el conjunto de prueba. La mayor IoU corresponde a #mejor("iou"), el mayor coeficiente de Dice a #mejor("dice"), la mayor precisión a #mejor("precision") y la mayor exhaustividad a #mejor("recall"). #todo[Mencionar la magnitud de la diferencia frente a la línea base y si es mayor que la desviación estándar entre particiones.]

#tabla-resultados() <tab:resultados>

== Resultados cualitativos <sec:res-cualitativos>

#todo[Describir la @fig:cualitativo: qué se observa en aciertos y en fallos. Incluir al menos un caso donde los modelos fallen.]

#figura(
  fig-pendiente(alto: 7cm, texto: "Cuadrícula: imagen, máscara de referencia y predicción de cada modelo"),
  caption: [Ejemplos cualitativos de segmentación. #todo[Indicar el orden de columnas y qué casos se muestran.]],
  completo: true,
) <fig:cualitativo>

== Estudio de ablación <sec:res-ablacion>

#todo[Efecto de cada decisión de diseño: codificador congelado frente a ajustado, entrada de 3 frente a 64 dimensiones, tamaño del conjunto de entrenamiento. Presentar con una tabla o figura adicional.]
