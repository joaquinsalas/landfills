# Compilación reproducible del artículo.
# Requiere el CLI de Typst (versión en .typst-version). Usa solo las fuentes
# incluidas en Typst (--ignore-system-fonts) para que el PDF sea idéntico
# en cualquier máquina.

TYPST ?= typst
FLAGS := --ignore-system-fonts
SRC   := main.typ
OUT   := build

.PHONY: borrador final watch version clean

borrador: version
	mkdir -p $(OUT)
	$(TYPST) compile $(FLAGS) $(SRC) $(OUT)/articulo-borrador.pdf

# Falla si queda algún marcador «REEMPLAZAR», figura pendiente o dato de ejemplo.
final: version
	mkdir -p $(OUT)
	$(TYPST) compile $(FLAGS) --input draft=false $(SRC) $(OUT)/articulo.pdf

watch:
	mkdir -p $(OUT)
	$(TYPST) watch $(FLAGS) $(SRC) $(OUT)/articulo-borrador.pdf

version:
	@echo "Typst requerido: $$(cat .typst-version) | instalado: $$($(TYPST) --version)"

clean:
	rm -rf $(OUT)
