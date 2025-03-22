# Makefile for DAPS paper

# Main paper file
MAIN = daps_paper

# Output directory
OUTPUT_DIR = build

# Command for running pdflatex with options
PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(OUTPUT_DIR)

# Default target
all: setup $(OUTPUT_DIR)/$(MAIN).pdf

# Create output directory
setup:
	mkdir -p $(OUTPUT_DIR)

# Build the PDF file
$(OUTPUT_DIR)/$(MAIN).pdf: $(MAIN).tex arxiv.sty
	$(PDFLATEX) $(MAIN).tex
	$(PDFLATEX) $(MAIN).tex  # Run twice for references

# Clean the output directory
clean:
	rm -rf $(OUTPUT_DIR)

# View the PDF (adjust based on your PDF viewer)
view: $(OUTPUT_DIR)/$(MAIN).pdf
	xdg-open $(OUTPUT_DIR)/$(MAIN).pdf || open $(OUTPUT_DIR)/$(MAIN).pdf || start $(OUTPUT_DIR)/$(MAIN).pdf

.PHONY: all setup clean view 