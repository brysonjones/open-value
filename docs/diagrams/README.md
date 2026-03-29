# Building The Architecture Diagram

This directory contains the source file `architecture.tex` and the final image `architecture.png` used by the docs.

## Prerequisites

- A LaTeX toolchain (`pdflatex` or `latexmk`)
- Poppler (`pdftoppm`) or ImageMagick (`magick`)

## Build Steps

Run from this directory:

```bash
cd /home/bjones/ws/open-value/docs/diagrams

# 1) Compile LaTeX to PDF
latexmk -pdf -interaction=nonstopmode architecture.tex

# 2) Convert first PDF page to PNG in this same directory
pdftoppm -png -singlefile architecture.pdf architecture
```

Then, `architecture.png` is updated in `docs/diagrams/`.
