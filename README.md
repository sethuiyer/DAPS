# DAPS - Dimensionally Adaptive Prime Search

<img src="./LOGO.webp" height="256px" width="256px" alt="DAPS Logo" align="right"/>

A high-performance global optimization algorithm for 1D, 2D, and 3D functions, implemented in C++ with Python bindings.

## Overview

DAPS efficiently finds global minima of complex functions using a prime number-based adaptive grid search strategy. It excels at navigating complex landscapes with multiple local minima, valleys, and discontinuities.

### Key Features

- **Multi-Dimensional**: Optimize functions in 1D, 2D, or 3D spaces
- **High Performance**: C++ core with Cython bindings
- **Global Optimization**: Designed to escape local minima
- **Adaptive Resolution**: Dynamically adjusts search precision
- **SciPy Compatible**: Familiar interface for easy integration

## Quick Start

```bash
# Install from PyPI
pip install daps

# Or install from source
git clone https://github.com/sethuiyer/DAPS.git
cd DAPS
pip install -e .
```

## Basic Usage

```python
from daps import daps_minimize

# 1D Optimization Example
result = daps_minimize(
    'sphere_function',
    bounds=[-5, 5],
    options={'dimensions': 1, 'maxiter': 50}
)

print(f"Optimal solution: {result['x']}, value: {result['fun']}")
```

## Custom Functions

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

# Define a custom 2D function
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Wrap in a DAPSFunction with metadata
func = DAPSFunction(
    func=himmelblau,
    name="Himmelblau",
    bounds=[-5, 5, -5, 5],
    dimensions=2,
    description="Classic test function with four identical local minima"
)

# Optimize
result = daps_minimize(func, options={'maxiter': 80})
print(f"Optimal point: ({result['x'][0]:.4f}, {result['x'][1]:.4f})")
```

## ⚠️ Development Status

The pure Python implementation (`base.py`) is fully functional. C++/Cython integration and packaging are under active development.

## Interactive Demo

```bash
cd interactive
./run_demo.sh  # Linux/Mac
# or
run_demo.bat   # Windows
```

## Documentation

Full documentation: [https://sethuiyer.github.io/DAPS/](https://sethuiyer.github.io/DAPS/)

## How It Works

DAPS uses prime number-based grid sampling to avoid aliasing problems common in regular grid search methods. It dynamically adapts resolution and shrinks the search domain around promising regions.

For theoretical details, see the [research paper](paper/build/daps_paper.pdf).

## Citation

```bibtex
@article{iyerpreprintprime,
  title={Prime-Adaptive Search (PAS): A Novel Method for Efficient Optimization in Discontinuous Landscapes},
  author={Iyer, Sethu},
  year={2025},
  url={https://github.com/sethuiyer/DAPS},
}
```

## License

MIT License - See LICENSE file for details.
