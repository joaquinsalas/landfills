#import "template.typ": *

// ---------------------------------------------------------------------
// 1. METADATOS  (portada, encabezados y propiedades del PDF)
// ---------------------------------------------------------------------
#let meta = (
  // Revisar o ajustar los dos títulos.
  titulo: "Detección y predicción de rellenos sanitarios en México mediante visión artificial: comparación de arquitecturas y modelos de fundación",
  titulo-corto: "Detección de rellenos sanitarios en México mediante visión artificial",
  titulo-en: "Landfill Detection and Prediction in Mexico Using Computer Vision: A Comparison of Architectures and Foundation Models",

  // Un diccionario por autor. «afil» = números de afiliación (ver abajo).
  // Marca «corresponsal: true» al autor de contacto.
  autores: (
    (
      nombre: "Emilio REEMPLAZAR-apellidos",
      afil: (1,),
      corresponsal: true,
      correo: "REEMPLAZAR-correo@dominio.mx",
    ),
    (nombre: "REEMPLAZAR: coautor o asesor (borrar si no aplica)", afil: (1,)),
  ),

  afiliaciones: (
    "REEMPLAZAR: Departamento o carrera, Instituto Tecnológico de Querétaro, Querétaro, México",
  ),

  resumen: [
    #todo[Contexto y problema en 1–2 frases: por qué importa inventariar rellenos sanitarios en México.]
    #todo[Método: qué modelos y qué datos se compararon.]
    #todo[Resultados clave con cifras: «la configuración X alcanzó IoU = 0.xx frente a 0.yy de la línea base».]
    #todo[Conclusión e implicación práctica. Objetivo total: 150–250 palabras.]
  ],
  abstract: [
    #todo[English abstract: same four moves (context, method, key results with numbers, conclusion).]
  ],
  palabras-clave: (
    "rellenos sanitarios",
    "visión artificial",
    "segmentación semántica",
    "percepción remota",
    "modelos de fundación",
    "DINOv3",
    "AlphaEarth",
  ),
  keywords: (
    "landfills",
    "computer vision",
    "semantic segmentation",
    "remote sensing",
    "foundation models",
    "DINOv3",
    "AlphaEarth",
  ),

  // Fecha FIJA: garantiza que el PDF sea idéntico en cada compilación.
  // Actualizar manualmente al enviar el manuscrito.
  fecha: datetime(year: 2026, month: 10, day: 16),

  papel: "us-letter", // "a4" para formato europeo
)

// ---------------------------------------------------------------------
// 2. ESTRUCTURA  (orden de las secciones; cada una es un archivo)
//    Para reordenar, quitar o agregar una sección, edita esta lista.
// ---------------------------------------------------------------------
#let secciones = (
  "01-introduccion",
  "02-trabajo-relacionado",
  "03-datos",
  "04-metodologia",
  "05-resultados",
  "06-discusion",
  "07-conclusiones",
  "08-declaraciones", // disponibilidad de código/datos, autoría, conflictos
  "09-apendice", // opcional
)

// ---------------------------------------------------------------------
// 3. ENSAMBLADO  (no suele requerir cambios)
// ---------------------------------------------------------------------
#show: articulo.with(..meta)

#for s in secciones {
  include "secciones/" + s + ".typ"
}

#bibliography("references.bib")
