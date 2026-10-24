// =====================================================================
//  template.typ — Diseño y utilidades del artículo
//
//  Editar este archivo SOLO para cambiar el diseño. El contenido va en:
//    main.typ        metadatos y estructura
//    secciones/*.typ texto de cada sección
//    data/*.csv      tablas dinámicas (modelos y resultados)
//    references.bib  bibliografía
// =====================================================================

// ---------------------------------------------------------------------
// 1. MODO DE COMPILACIÓN
//    borrador (por defecto): muestra marcadores, guías y marca de agua.
//    final (--input draft=false): oculta guías y ABORTA si queda algún
//    marcador «REEMPLAZAR» o datos de ejemplo.
// ---------------------------------------------------------------------
#let draft = sys.inputs.at("draft", default: "true") != "false"

#let _rojo = rgb("#9b1c1c")
#let _amarillo = rgb("#fff1a8")
#let _azul = rgb("#1b4f8a")

/// Marcador de texto pendiente. Sustituir la llamada completa por el texto real.
#let todo(body) = {
  if not draft {
    panic("Modo final: queda al menos un marcador #todo[...]. Reemplázalo o compila en borrador.")
  }
  highlight(
    fill: _amarillo,
    extent: 1.5pt,
    text(fill: _rojo, weight: "regular", style: "normal")[«REEMPLAZAR: #body»],
  )
}

/// Nota de redacción para el autor. Visible solo en borrador; se borra sola en final.
#let guia(body) = if draft {
  block(
    width: 100%,
    fill: luma(240),
    stroke: (left: 2pt + luma(120)),
    inset: (x: 8pt, y: 6pt),
    radius: (right: 2pt),
  )[
    #set text(size: 8pt, fill: luma(70), style: "italic", weight: "regular")
    #set par(justify: false)
    *Guía de redacción.* #body
  ]
}

/// Resalta cualquier cadena que contenga la palabra REEMPLAZAR (uso interno).
#let _txt(x) = {
  if type(x) == str and x.contains("REEMPLAZAR") {
    if not draft { panic("Texto sin reemplazar: " + x) }
    highlight(fill: _amarillo, extent: 1.5pt, text(fill: _rojo, x))
  } else { x }
}

// ---------------------------------------------------------------------
// 2. DATOS DINÁMICOS (se leen de data/)
// ---------------------------------------------------------------------
#let _estado = toml("data/estado.toml")
#let datos-ejemplo = _estado.at("datos-ejemplo", default: true)
#let modelos = csv("data/modelos.csv", row-type: dictionary)
#let resultados = csv("data/resultados.csv", row-type: dictionary)
#let n-modelos = modelos.len()

/// Formatea un número con `d` decimales fijos (punto decimal).
#let _fmt(x, d: 3) = {
  let s = str(calc.round(x, digits: d))
  if not s.contains(".") {
    s + "." + "0" * d
  } else {
    s + "0" * (d - s.split(".").at(1).len())
  }
}

/// Texto «M1 (0.812)» con el mejor modelo según la métrica `clave`
/// (iou, dice, precision, recall), calculado desde data/resultados.csv.
#let mejor(clave) = {
  let fila = resultados.sorted(key: f => -float(f.at(clave + "_media"))).first()
  [#fila.id (#_fmt(float(fila.at(clave + "_media"))))]
}

#let _metricas = (
  (clave: "iou", nombre: "IoU"),
  (clave: "dice", nombre: "Dice"),
  (clave: "precision", nombre: "Precisión"),
  (clave: "recall", nombre: "Recall"),
)

// ---------------------------------------------------------------------
// 3. TABLAS Y FIGURAS
// ---------------------------------------------------------------------

/// Tabla genérica estilo «booktabs». Devuelve un `figure` (acepta etiqueta detrás).
#let tabla(
  encabezado,
  filas,
  caption: none,
  alineacion: auto,
  columnas: auto,
  completo: false,
  tam: 8.5pt,
  nota: none,
) = figure(
  kind: table,
  placement: auto,
  scope: if completo { "parent" } else { "column" },
  caption: caption,
  {
    set text(size: tam)
    if nota != none { align(left, nota); v(2pt) }
    table(
      columns: columnas,
      align: alineacion,
      stroke: none,
      inset: (x: 5pt, y: 3.5pt),
      table.hline(stroke: 0.9pt),
      table.header(..encabezado.map(strong)),
      table.hline(stroke: 0.5pt),
      ..filas.flatten(),
      table.hline(stroke: 0.9pt),
    )
  },
)

/// Tabla de configuraciones, generada desde data/modelos.csv.
#let tabla-modelos(caption: [Configuraciones de segmentación evaluadas.]) = tabla(
  ([ID], [Codificador], [Decodificador], [Entrada], [Congelado]),
  modelos.map(m => (
    strong(m.id),
    _txt(m.codificador),
    _txt(m.decodificador),
    _txt(m.entrada),
    _txt(m.congelado),
  )),
  caption: caption,
  columnas: (auto, 1fr, 1fr, 1.2fr, auto),
  alineacion: (left, left, left, left, center),
  completo: true,
)

/// Tabla de resultados (media ± desviación), generada desde data/resultados.csv.
/// Resalta en negrita el mejor valor de cada métrica.
#let tabla-resultados(
  caption: [Desempeño en el conjunto de prueba (media #sym.plus.minus desviación estándar). En negrita, el mejor valor por métrica.],
) = {
  if datos-ejemplo and not draft {
    panic("data/resultados.csv contiene datos de ejemplo. Reemplázalos y pon datos-ejemplo = false en data/estado.toml.")
  }
  let mejores = _metricas.map(m => calc.max(..resultados.map(f => float(f.at(m.clave + "_media")))))
  let filas = resultados.map(f => (
    strong(f.id),
    .._metricas.enumerate().map(((i, m)) => {
      let v = float(f.at(m.clave + "_media"))
      let s = float(f.at(m.clave + "_std"))
      let celda = [#_fmt(v) #sym.plus.minus #_fmt(s)]
      if v == mejores.at(i) { strong(celda) } else { celda }
    }),
  ))
  tabla(
    (([Modelo],) + _metricas.map(m => [#m.nombre #sym.arrow.t])),
    filas,
    caption: caption,
    columnas: (auto,) + (1fr,) * 4,
    alineacion: (left,) + (center,) * 4,
    completo: true,
    nota: if datos-ejemplo {
      text(fill: _rojo, weight: "bold")[DATOS DE EJEMPLO: reemplazar data/resultados.csv y poner datos-ejemplo = false en data/estado.toml.]
    },
  )
}

/// Figura con comportamiento uniforme (acepta etiqueta detrás).
#let figura(cuerpo, caption: none, completo: false) = figure(
  cuerpo,
  caption: caption,
  placement: auto,
  scope: if completo { "parent" } else { "column" },
)

/// Recuadro gris que ocupa el lugar de una figura. Reemplazar por
/// `image("figures/archivo.pdf", width: 100%)`.
#let fig-pendiente(alto: 4.5cm, texto: "Figura pendiente") = {
  if not draft { panic("Modo final: queda una figura pendiente (fig-pendiente).") }
  rect(
    width: 100%,
    height: alto,
    fill: luma(238),
    stroke: (paint: luma(150), dash: "dashed", thickness: 0.6pt),
  )[#align(center + horizon, text(fill: _rojo, size: 8.5pt, weight: "bold")[«REEMPLAZAR: #texto»])]
}

// ---------------------------------------------------------------------
// 4. DISEÑO DEL ARTÍCULO
// ---------------------------------------------------------------------
#let articulo(
  titulo: "",
  titulo-corto: none,
  titulo-en: none,
  autores: (),
  afiliaciones: (),
  resumen: none,
  abstract: none,
  palabras-clave: (),
  keywords: (),
  fecha: none,
  papel: "us-letter",
  fuente: "New Computer Modern",
  tam: 10pt,
  doc,
) = {
  // --- Metadatos del PDF (fecha fija => PDF reproducible) ---
  set document(
    title: titulo,
    author: autores.map(a => a.nombre),
    keywords: palabras-clave,
    date: fecha,
  )

  // --- Página ---
  set page(
    paper: papel,
    margin: (x: 1.9cm, top: 2.3cm, bottom: 2.2cm),
    columns: 2,
    background: if draft {
      place(center + horizon, rotate(-35deg, text(size: 96pt, weight: "bold", fill: luma(240))[BORRADOR]))
    },
    header: context {
      if counter(page).get().first() > 1 {
        set text(size: 8pt, fill: luma(80))
        stack(
          spacing: 3pt,
          align(center, emph(if titulo-corto != none { titulo-corto } else { titulo })),
          line(length: 100%, stroke: 0.4pt + luma(160)),
        )
      }
    },
    footer: context align(center, text(size: 8pt, fill: luma(80), counter(page).display("1 de 1", both: true))),
  )
  set columns(gutter: 1.3em)

  // --- Texto y párrafos ---
  set text(font: fuente, size: tam, lang: "es", region: "MX")
  set par(justify: true, leading: 0.6em, spacing: 0.75em)
  set enum(indent: 0.6em, body-indent: 0.6em, spacing: 0.5em)
  set list(indent: 0.6em, body-indent: 0.6em, spacing: 0.5em)
  show link: set text(fill: _azul)
  show ref: set text(fill: _azul)

  // --- Encabezados ---
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => block(above: 1.5em, below: 0.8em, sticky: true)[
    #set text(size: 10pt, weight: "bold")
    #if it.numbering != none [#counter(heading).display(it.numbering)#h(0.7em)]#upper(it.body)
  ]
  show heading.where(level: 2): it => block(above: 1.2em, below: 0.6em, sticky: true)[
    #set text(size: 10pt, weight: "bold", style: "italic")
    #if it.numbering != none [#counter(heading).display(it.numbering)#h(0.6em)]#it.body
  ]
  show heading.where(level: 3): it => block(above: 1em, below: 0.5em, sticky: true)[
    #set text(size: 10pt, weight: "regular", style: "italic")
    #it.body
  ]

  // --- Ecuaciones ---
  set math.equation(numbering: "(1)", supplement: [Ec.])
  show math.equation: set block(spacing: 0.8em)

  // --- Figuras y tablas ---
  set figure.caption(separator: [. ])
  show figure.caption: set text(size: 8.5pt)
  show figure.caption: set par(justify: true)
  show figure.caption: it => [#strong[#it.supplement #it.counter.display(it.numbering)#it.separator]#it.body]
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.where(kind: table): set block(breakable: false)

  // --- Bibliografía ---
  set bibliography(style: "ieee", title: "Referencias")
  show bibliography: set text(size: 8.5pt)
  show bibliography: set par(justify: false)

  // --- Bloque de título (a todo el ancho, primera página) ---
  let marca-autor(a) = {
    let marcas = a.at("afil", default: ()).map(str).join(",")
    if marcas == none { marcas = "" }
    if a.at("corresponsal", default: false) { marcas += "*" }
    [#_txt(a.nombre)#super(marcas)]
  }
  let corresponsales = autores.filter(a => a.at("corresponsal", default: false))

  let bloque-titulo = block(width: 100%)[
    #set par(justify: false)
    #align(center)[
      #text(size: 17pt, weight: "bold")[#titulo]
      #if titulo-en != none [
        #v(0.2em)
        #text(size: 11.5pt, style: "italic")[#titulo-en]
      ]
      #v(0.8em)
      #text(size: 10.5pt)[#autores.map(marca-autor).join(", ", last: " y ")]
      #v(0.4em)
      #set text(size: 8.5pt)
      #afiliaciones.enumerate().map(((i, a)) => [#super[#(i + 1)]#_txt(a)]).join(linebreak())
      #if corresponsales.len() > 0 [
        #v(0.2em)
        *Correspondencia:* #corresponsales.map(a => [#_txt(a.nombre) (#_txt(a.at("correo", default: "")))]).join("; ")
      ]
    ]
    #v(0.9em)
    #block(width: 100%, inset: (x: 1.1cm))[
      #set text(size: 9pt)
      #set par(justify: true)
      #if resumen != none [
        *Resumen.* #resumen
        #v(0.3em)
        *Palabras clave:* #palabras-clave.join("; ").
      ]
      #if abstract != none [
        #v(0.6em)
        *Abstract.* #abstract
        #v(0.3em)
        *Keywords:* #keywords.join("; ").
      ]
    ]
    #v(0.4em)
    #line(length: 100%, stroke: 0.5pt)
  ]

  place(top + center, scope: "parent", float: true, clearance: 1.4em, bloque-titulo)

  doc
}
