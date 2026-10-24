#import "../template.typ": *

= Metodología <sec:metodos>

#guia[Describe todo lo necesario para reproducir los experimentos y justifica cada decisión de diseño (por ejemplo, por qué se congelan los codificadores). Las tablas se generan desde data/modelos.csv: edita ese archivo, no el texto.]

== Arquitecturas evaluadas <sec:metodos-modelos>

Se evalúan #n-modelos configuraciones (@tab:modelos). El modelo M3, una U-Net estándar entrenada sin preentrenamiento, actúa como línea base. #todo[Verificar que data/modelos.csv refleje exactamente las pruebas del repositorio y agregar la justificación de cada configuración.]

#tabla-modelos() <tab:modelos>

== Estrategia de entrenamiento <sec:metodos-entrenamiento>

Los codificadores preentrenados (DINOv3 y ResNet) se mantienen congelados para preservar el conocimiento adquirido en el preentrenamiento y reducir el riesgo de sobreajuste con un conjunto etiquetado pequeño; solo se entrenan los módulos nuevos. #todo[Función de pérdida y su justificación, optimizador, tasa de aprendizaje, programación, número de épocas, tamaño de lote, criterio de parada.]

#tabla(
  ([Parámetro], [Valor]),
  (
    ([Optimizador], todo[valor]),
    ([Tasa de aprendizaje], todo[valor]),
    ([Épocas], todo[valor]),
    ([Tamaño de lote], todo[valor]),
    ([Función de pérdida], todo[valor]),
    ([Tamaño de entrada], todo[valor]),
    ([Semillas aleatorias], todo[valor]),
    ([Hardware], todo[GPU y memoria]),
  ),
  caption: [Hiperparámetros comunes a todas las configuraciones.],
  columnas: (1fr, 1fr),
  alineacion: (left, left),
) <tab:hiperparametros>

== Métricas de evaluación <sec:metodos-metricas>

Sean $P$ la máscara predicha y $G$ la máscara de referencia. Se reportan la intersección sobre unión (IoU), el coeficiente de Dice, la precisión y la exhaustividad (_recall_):

$ "IoU" = (|P inter G|) / (|P union G|) $ <eq:iou>

$ "Dice" = (2 |P inter G|) / (|P| + |G|) $ <eq:dice>

$ "Precisión" = "TP" / ("TP" + "FP"), quad "Recall" = "TP" / ("TP" + "FN") $ <eq:pr>

donde TP, FP y FN son los píxeles verdaderos positivos, falsos positivos y falsos negativos, respectivamente. #todo[Indicar la métrica principal para comparar modelos y por qué (la IoU es habitual cuando la clase positiva es minoritaria).]

== Protocolo experimental <sec:metodos-protocolo>

#todo[Número de particiones y semillas, cómo se calcula la media y la desviación estándar, y qué garantiza una comparación justa (mismos datos, mismas particiones, mismo presupuesto de entrenamiento para todos los modelos).]

== Reproducibilidad <sec:metodos-repro>

#todo[Versiones de software (Python, PyTorch, bibliotecas de segmentación), semillas, hardware y enlace al repositorio con código y configuraciones.]
