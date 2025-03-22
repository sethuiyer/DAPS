# DAPS Research Paper

This directory contains the LaTeX source for the DAPS (Dimensionally Adaptive Prime Search) research paper.

## Paper Title

"Prime-Adaptive Search (PAS): A Novel Method for Efficient Optimization in Discontinuous Landscapes"

## Contents

- `daps_paper.tex`: Main LaTeX source file for the paper
- `arxiv.sty`: Style file for arXiv formatting
- `Makefile`: Build system for compiling the paper

## Building the Paper

### Prerequisites

You need a LaTeX distribution installed on your system (e.g., TeX Live, MiKTeX).

### Compilation Instructions

To compile the paper, you can use the provided Makefile:

```bash
# Build the PDF
make

# View the PDF (opens with your default PDF viewer)
make view

# Clean build artifacts
make clean
```

Alternatively, you can compile manually with:

```bash
# Create build directory
mkdir -p build

# Run pdflatex (twice to ensure references are resolved)
pdflatex -output-directory=build daps_paper.tex
pdflatex -output-directory=build daps_paper.tex
```

The compiled PDF will be available at `build/daps_paper.pdf`.

## Paper Abstract

Modern optimization problems increasingly involve discontinuous, non-smooth, and multi-modal functions, rendering traditional gradient-based methods ineffective. This paper introduces Prime-Adaptive Search (PAS), a novel iterative optimization technique that leverages prime number properties to adaptively refine the search space. PAS employs a prime-driven partitioning scheme to avoid aliasing and to naturally provide a hierarchical resolution of the search domain. By focusing on function evaluations at prime-partitioned grids and adaptively shrinking the domain around promising candidates, PAS excels in robustly handling non-smooth functions and navigating multi-modal landscapes. We present empirical results from benchmark problems, including discontinuous functions, LeetCode-style "peak-finding," and challenging 2D/3D scenarios. Our findings demonstrate PAS's advantages—adaptive resolution, avoidance of periodic sampling bias, and gradient independence—all culminating in strong performance for a broad range of practical applications, from AI hyperparameter tuning to cryptographic parameter searches. 