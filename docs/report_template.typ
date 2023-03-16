#let project(title: "", abstract: [], keywords: [], authors: (), body) = {
  // Set the document's basic properties.
  set document(author: authors.map(a => a.name), title: title)
  set page(numbering: "1", number-align: center, paper: "us-letter")
  set text(font: "Linux Libertine", lang: "en")
  set heading(numbering: "1.1.")

  // Title row.
  align(center)[
    #block(text(weight: 700, 1.75em, title))
  ]

  // Author information.
  pad(
    top: 0.5em,
    bottom: 0.5em,
    x: 2em,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(center)[
        *#author.name* \
        #emph[#author.affl] \
        #link("mailto:"+author.email)
      ]),
    ),
  )

  // Main body.
  set par(justify: true)

  align(center)[
    #heading(outlined: false, numbering: none, text(0.85em, smallcaps[Abstract]))
  ]
  pad(
    left: 2em,
    right: 2em,
    abstract +
    [\ *Keywords:* ] + keywords
  )

  body
}