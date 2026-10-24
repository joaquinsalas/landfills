# Plantilla de artículo científico en Typst

Detección de rellenos sanitarios en México mediante visión artificial.
Probada con Typst 0.15.1. Idioma: español. Formato: dos columnas, estilo de citas IEEE.

## Estructura

```
main.typ                  metadatos + orden de secciones (empieza aquí)
template.typ              diseño y utilidades (no editar salvo cambio de diseño)
secciones/01..09-*.typ    texto de cada sección
data/modelos.csv          configuraciones evaluadas  -> tabla automática
data/resultados.csv       métricas por modelo        -> tabla y frases automáticas
data/estado.toml          interruptor de datos de ejemplo
references.bib            bibliografía
figures/                  imágenes (PDF o SVG recomendado)
Makefile                  compilación reproducible
```

## Compilar

```
make            borrador: marcadores amarillos, guías grises, marca de agua
make final      versión limpia; ABORTA si queda algo sin reemplazar
make watch      recompila al guardar
```

## Qué reemplazar

1. Todo texto amarillo «REEMPLAZAR: ...» en el PDF. En el código aparece como
   `#todo[...]` (sustituye la llamada completa por tu texto) o como una cadena
   que contiene la palabra REEMPLAZAR.
2. `main.typ`: autores, afiliaciones, correo, resumen, abstract. Revisar títulos y palabras clave.
3. `data/modelos.csv`: verificar contra las pruebas del repositorio (columnas con REEMPLAZAR).
4. `data/resultados.csv`: pegar métricas reales (media y desviación por modelo)
   y poner `datos-ejemplo = false` en `data/estado.toml`.
5. Figuras: sustituir cada `fig-pendiente(...)` por `image("../figures/archivo.pdf", width: 100%)`.
6. `references.bib`: agregar las fuentes reales y citar con `@clave`.
7. Borrar las guías `#guia[...]` si ya no las necesitas (desaparecen solas en `make final`).

## Comportamiento dinámico

- Las tablas de modelos y resultados se generan desde `data/*.csv`; el mejor valor
  de cada métrica sale en negrita.
- Frases como «la mayor IoU corresponde a M1 (0.900)» se calculan solas desde el CSV.
- El número de modelos en el texto (`#n-modelos`) sale del CSV.
- Referencias cruzadas (`@sec:datos`, `@tab:resultados`, `@fig:cualitativo`, `@eq:iou`)
  se renumeran solas.
- Para reordenar o quitar secciones, edita la lista `secciones` en `main.typ`.

## Reproducibilidad

- Versión de Typst fijada en `.typst-version`.
- `--ignore-system-fonts`: solo fuentes incluidas en Typst (New Computer Modern).
- Fecha del documento fija en `main.typ` (no usa la fecha del sistema).
- Mismos archivos de entrada producen el mismo PDF.

## Antes de enviar

- Confirmar que `make final` compila sin errores.
- Confirmar que el destino acepta Typst; muchas revistas exigen LaTeX o Word.
  En ese caso, `main.typ` sirve como borrador y el contenido se migra.
