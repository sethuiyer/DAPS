#!/bin/bash
set -e  # Exit on error

echo "=== Building DAPS paper ==="
echo "1. Setting up environment"

# Check if virtual environment exists, create if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -q -U pip
pip install -q -r requirements.txt

# Try to install DAPS in dev mode, but don't fail if it doesn't work
echo "Attempting to install DAPS package (optional)..."
cd ..
pip install -q -e . || {
    echo "  Warning: Could not install DAPS package in development mode."
    echo "  This is ok if you're just building the paper."
}
cd paper

echo "2. Generating assets..."
# Create figures directory if it doesn't exist
mkdir -p figures
mkdir -p ../docs/assets

# Generate DAPS logo first
echo "  - Generating DAPS logo..."
python generate_logo.py || {
    echo "  Warning: Could not generate DAPS logo. Continuing anyway."
}

# Generate other figures
echo "  - Generating figures for paper..."
python generate_figures.py || {
    echo "  Warning: Could not generate figures. Continuing with LaTeX compilation."
}

echo "3. Compiling LaTeX paper..."
# Create build directory
mkdir -p build

# Check if the LaTeX document has the correct natbib configuration
if ! grep -q "\\usepackage\[numbers\]{natbib}" daps_paper.tex; then
    echo "Fixing natbib package configuration for numerical citations..."
    sed -i 's/\\usepackage{natbib}/\\usepackage[numbers]{natbib}/g' daps_paper.tex
fi

# Check if algorithmicx is included
if ! grep -q "\\usepackage{algorithmicx}" daps_paper.tex; then
    echo "Adding required algorithmicx package..."
    sed -i 's/\\usepackage{algorithm}/\\usepackage{algorithm}\n\\usepackage{algorithmicx}/g' daps_paper.tex
fi

# Run pdflatex twice to resolve references
echo "  - First LaTeX pass..."
pdflatex -interaction=nonstopmode -output-directory=build daps_paper.tex || {
    echo "LaTeX encountered errors on first pass. Checking log file..."
    grep -A 3 -B 3 "Error" build/daps_paper.log || echo "No specific error found in log."
    echo "Please check build/daps_paper.log for details."
    exit 1
}

echo "  - Second LaTeX pass to resolve references..."
pdflatex -interaction=nonstopmode -output-directory=build daps_paper.tex || {
    echo "LaTeX encountered errors on second pass."
    exit 1
}

echo "4. Copying paper to docs..."
# Ensure figures are incorporated in the paper
mkdir -p ../docs/assets
cp build/daps_paper.pdf ../docs/assets/

echo "=== Paper build complete ==="
echo "PDF available at: build/daps_paper.pdf"
echo "PDF also copied to: ../docs/assets/daps_paper.pdf"

# Optional: open the PDF if on a compatible system
if command -v xdg-open &> /dev/null; then
    xdg-open build/daps_paper.pdf &>/dev/null &  # Run in background and suppress output
elif command -v open &> /dev/null; then
    open build/daps_paper.pdf &>/dev/null &
elif command -v start &> /dev/null; then
    start build/daps_paper.pdf
else
    echo "Please open the PDF manually"
fi

# Deactivate virtual environment
deactivate 