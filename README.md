# DAPS - Dimensionally Adaptive Prime Search

![img](LOGO.webp)


A high-performance global optimization algorithm for 1D, 2D, and 3D functions, implemented in C++ with Python bindings via Cython.

## Overview

DAPS (Dimensionally Adaptive Prime Search) is a novel optimization algorithm designed for efficiently finding global minima of complex functions in 1D, 2D, and 3D spaces. It utilizes a unique prime number-based grid search strategy with adaptive refinement to navigate complex objective function landscapes with multiple local minima, valleys, and cliffs.

### Key Features

- **Multi-Dimensional Support**: Optimize functions in 1D, 2D, or 3D spaces
- **High Performance**: C++ core with Cython bindings for speed
- **Global Optimization**: Designed to escape local minima and find global optima
- **Advanced Adaptivity**: Dynamically adjusts search resolution in each dimension
- **Built-in Test Functions**: Includes several challenging benchmark functions
- **Custom Function Support**: Easily define and optimize your own functions
- **Comprehensive API**: Simple interface with detailed output and callback support
- **SciPy Compatible**: Similar interface to SciPy's optimization functions
- **Interactive Demo**: Visualize the optimization process using the Streamlit app

## Pythonic Implementation

Python implementation can be found in `base.py`

### Run Locally

```bash
# Clone the repository
git clone https://github.com/sethuiyer/DAPS.git
cd DAPS

# Install DAPS
pip install -e .

# Run the demo
cd interactive
./run_demo.sh  # Linux/Mac
# or
run_demo.bat   # Windows
```

## Research Paper

The algorithm is described in detail in the research paper:

**"Prime-Adaptive Search (PAS): A Novel Method for Efficient Optimization in Discontinuous Landscapes"**

### Abstract

Modern optimization problems increasingly involve discontinuous, non-smooth, and multi-modal functions, rendering traditional gradient-based methods ineffective. This paper introduces Prime-Adaptive Search (PAS), a novel iterative optimization technique that leverages prime number properties to adaptively refine the search space. PAS employs a prime-driven partitioning scheme to avoid aliasing and to naturally provide a hierarchical resolution of the search domain. By focusing on function evaluations at prime-partitioned grids and adaptively shrinking the domain around promising candidates, PAS excels in robustly handling non-smooth functions and navigating multi-modal landscapes. We present empirical results from benchmark problems, including discontinuous functions, LeetCode-style "peak-finding," and challenging 2D/3D scenarios. Our findings demonstrate PAS's advantages—adaptive resolution, avoidance of periodic sampling bias, and gradient independence—all culminating in strong performance for a broad range of practical applications, from AI hyperparameter tuning to cryptographic parameter searches.

To compile the LaTeX paper:

```bash
cd paper
./build_paper.sh
```

The compiled paper will be available at `paper/build/daps_paper.pdf`.

## Installation

### From PyPI (Recommended)

```bash
pip install daps
```

### From Source

```bash
git clone https://github.com/sethuiyer/DAPS.git
cd DAPS
pip install -e .
```

This will compile the C++ code with Cython and install the package in development mode.

## Requirements

- Python 3.6+
- NumPy
- Cython (for building from source)
- A C++ compiler with C++11 support

## Basic Usage

```python
from daps import daps_minimize

# 1D Optimization Example
result_1d = daps_minimize(
    'sphere_function',
    bounds=[-5, 5],  # 1D bounds: [x_min, x_max]
    options={'dimensions': 1, 'maxiter': 50}
)

# 2D Optimization Example
result_2d = daps_minimize(
    'recursive_fractal_cliff_valley',
    bounds=[-5, 5, -5, 5],  # 2D bounds: [x_min, x_max, y_min, y_max]
    options={'dimensions': 2, 'maxiter': 80}
)

# 3D Optimization Example
result_3d = daps_minimize(
    'recursive_fractal_cliff_valley',
    bounds=[-5, 5, -5, 5, -5, 5],  # 3D bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
    options={'dimensions': 3, 'maxiter': 100}
)

print(f"1D Best solution: {result_1d['x']}, value: {result_1d['fun']}")
print(f"2D Best solution: {result_2d['x']}, value: {result_2d['fun']}")
print(f"3D Best solution: {result_3d['x']}, value: {result_3d['fun']}")
```

### Optimizing a Custom Function

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

# Define a custom 1D function
def parabola_1d(x):
    return (x - 2)**2

# Define a custom 2D function
def himmelblau_2d(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Define a custom 3D function
def my_function(x, y, z):
    return np.sin(x*y) + np.cos(y*z) + x**2 + y**2 + z**2

# Create DAPSFunction instances with metadata
parabola = DAPSFunction(
    func=parabola_1d,
    name="Parabola 1D",
    bounds=[-10, 10],
    dimensions=1,
    true_optimum=[2.0],
    true_value=0.0,
    description="Simple 1D parabola with minimum at x=2"
)

himmelblau = DAPSFunction(
    func=himmelblau_2d,
    name="Himmelblau 2D",
    bounds=[-5, 5, -5, 5],
    dimensions=2,
    true_optimum=[[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]],
    true_value=0.0,
    description="Himmelblau function with four identical local minima"
)

custom_3d = DAPSFunction(
    func=my_function,
    name="Custom 3D Function",
    bounds=[-5, 5, -5, 5, -5, 5],
    dimensions=3,
    description="A custom 3D function with multiple local minima"
)

# Optimize the functions
result_1d = daps_minimize(parabola, options={'maxiter': 50})
result_2d = daps_minimize(himmelblau, options={'maxiter': 80})
result_3d = daps_minimize(
    custom_3d,
    options={
        'maxiter': 200,
        'min_prime_idx': 5,
        'max_prime_idx': 15,
        'tol': 1e-6
    }
)

# Process the results
print(f"1D Optimal point: {result_1d['x'][0]:.4f}")
print(f"2D Optimal point: ({result_2d['x'][0]:.4f}, {result_2d['x'][1]:.4f})")
print(f"3D Optimal point: ({result_3d['x'][0]:.4f}, {result_3d['x'][1]:.4f}, {result_3d['x'][2]:.4f})")
```

## Documentation

Full documentation is available at [https://sethuiyer.github.io/DAPS/](https://sethuiyer.github.io/DAPS/).

The documentation includes:
- Installation instructions
- Getting started guide
- API reference
- Interactive demo guide
- Advanced configuration options
- Benchmark results
- Theoretical background

## Project Structure

```
daps/
├── __init__.py           # Main package interface
├── core/                 # Core implementation
│   ├── __init__.py       # Core package interface
│   ├── _daps.pyx         # Cython interface to C++
│   ├── daps.cpp          # C++ implementation
│   ├── function.py       # Function validation and interface
│   ├── optimizer.py      # Python optimizer interface
│   └── test_functions.py # Built-in test functions
│
tests/                    # Test suite
├── __init__.py
└── test_daps.py          # Comprehensive tests
│
examples/                 # Usage examples
├── basic_example.py      # Simple usage example
└── benchmark.py          # Benchmarking against other optimizers
│
interactive/              # Interactive demo
├── app.py                # Streamlit application
├── requirements.txt      # Demo-specific requirements
├── run_demo.sh           # Unix/Linux/macOS run script
└── run_demo.bat          # Windows run script
│
paper/                    # Research paper
├── daps_paper.tex        # LaTeX source for the paper
├── arxiv.sty             # Style file for arXiv format
├── build_paper.sh        # Script for building the paper
└── generate_figures.py   # Script to generate paper figures
│
docs/                     # Documentation
├── index.md              # Home page
├── usage/                # Usage guides
├── research/             # Research materials
└── assets/               # Images and other assets
```

## API Reference

### `daps_minimize(func, bounds=None, options=None)`

Main optimization function with a SciPy-like interface.

**Parameters:**
- `func`: Function to minimize (callable, string name of built-in function, or DAPSFunction instance)
- `bounds`: Bounds for variables, with length depending on dimensions:
  - 1D: [x_min, x_max]
  - 2D: [x_min, x_max, y_min, y_max]
  - 3D: [x_min, x_max, y_min, y_max, z_min, z_max]
- `options`: Dictionary of options:
  - `dimensions`: Number of dimensions (1, 2, or 3). Auto-detected from bounds if not specified.
  - `maxiter`: Maximum number of iterations (default: 1000)
  - `min_prime_idx`: Minimum prime index (default: 5)
  - `max_prime_idx`: Maximum prime index (default: 20)
  - `callback`: Function called after each iteration (default: None)
  - `tol`: Tolerance for termination (default: 1e-8)

**Returns:**
- Dictionary containing:
  - `x`: Array of optimal values (length depends on dimensions)
  - `fun`: Function value at optimum
  - `nfev`: Number of function evaluations
  - `nit`: Number of iterations
  - `success`: Whether optimization succeeded
  - `dimensions`: Number of dimensions used for optimization
  - `final_prime_indices`: Final prime indices used

### `DAPSFunction`

Class for defining functions with metadata for optimization.

**Parameters:**
- `func`: The function to optimize (must accept arguments based on dimensions)
- `name`: Name of the function
- `bounds`: List of bounds, length depends on dimensions
- `dimensions`: Number of dimensions (1, 2, or 3)
- `true_optimum`: Known optimal point(s) (if available)
- `true_value`: Known optimal value (if available)
- `description`: Description of the function

## Built-in Test Functions

All built-in test functions can be used in 1D, 2D, or 3D modes:

- **Recursive Fractal Cliff Valley**: A challenging function with fractal-like structure
- **Rosenbrock**: Classic banana-shaped valley test function
- **Sphere Function**: Simple convex function
- **Ackley Function**: Highly non-convex function with many local minima
- **Rastrigin Function**: Highly multimodal function with many regular local minima

## License

MIT License - See LICENSE file for details.

## Citation

If you use DAPS in your research or projects, please consider citing:

```bibtex
@article{iyerpreprintprime,
  title={Prime-Adaptive Search (PAS): A Novel Method for Efficient Optimization in Discontinuous Landscapes},
  author={Iyer, Sethu},
  year={2025},
  url={https://github.com/sethuiyer/DAPS},
}
```

Or simply:

> Sethu Iyer (2025). Prime-Adaptive Search (PAS): A Novel Method for Efficient Optimization in Discontinuous Landscapes. GitHub repository: https://github.com/sethuiyer/DAPS

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
