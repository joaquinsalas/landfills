#import "../template.typ": *

// Los encabezados de este archivo se numeran como A, A.1, ... y se citan como «Apéndice A».
#set heading(numbering: "A.1", supplement: [Apéndice])
#counter(heading).update(0)

= Material suplementario <sec:apendice>

#guia[Opcional. Resultados detallados por partición, curvas de entrenamiento, tablas extensas. Quita `09-apendice` de la lista en main.typ si no lo necesitas.]

#todo[Contenido suplementario (resultados por partición, curvas de entrenamiento, configuraciones completas).]
