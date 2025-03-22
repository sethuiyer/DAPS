# DAPS: Dimensionally Adaptive Prime Search

<p align="center">
<img src="assets/daps_logo.png" alt="DAPS Logo" width="300">
</p>

<div class="badges" align="center">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/daps">
  <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/daps">
  <img alt="Tests" src="https://github.com/sethuiyer/DAPS/workflows/Tests/badge.svg">
  <img alt="Documentation" src="https://github.com/sethuiyer/DAPS/workflows/Documentation/badge.svg">
  <img alt="License" src="https://img.shields.io/github/license/sethuiyer/DAPS">
  <img alt="Downloads" src="https://pepy.tech/badge/daps">
</div>

## Overview

DAPS (Dimensionally Adaptive Prime Search) is a novel optimization algorithm specifically designed for challenging functions with discontinuities, multiple local minima, and non-convex landscapes. By leveraging prime number-based grid sampling and adaptive domain shrinking, DAPS effectively navigates complex optimization problems without requiring gradient information.

<div class="grid cards" markdown>

-   :material-speedometer:{ .lg .middle } __High Performance__

    ---

    DAPS excels at optimizing discontinuous and complex functions where gradient-based methods struggle.

-   :material-cube-scan:{ .lg .middle } __3D Specialized__

    ---

    Specifically designed for 3D optimization problems, with a focus on scientific and engineering applications.

-   :material-cog-refresh:{ .lg .middle } __No Gradients Required__

    ---

    Works with black-box functions where gradient information is unavailable or expensive to compute.

-   :material-language-python:{ .lg .middle } __Easy to Use__

    ---

    Simple Python API makes it straightforward to integrate DAPS into your existing projects.

</div>

## Key Features

- **Prime-based grid sampling**: Avoids aliasing effects common in regular grid search
- **Adaptive domain shrinking**: Efficiently narrows search space around promising regions
- **Gradient-free optimization**: No derivatives or gradient information required
- **Robust to discontinuities**: Handles functions with sharp transitions and discontinuities
- **Parallelizable**: Can leverage multi-core architectures for faster optimization
- **C++ core with Python interface**: Combines performance with ease of use

## Quick Start

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

# Define a 3D function to minimize
def my_function(x, y, z):
    return np.sin(x*y) + np.cos(y*z) + x**2 + y**2 + z**2

# Create a DAPSFunction instance
func = DAPSFunction(
    func=my_function,
    name="Custom Function",
    bounds=[-5, 5, -5, 5, -5, 5]
)

# Run the optimization
result = daps_minimize(
    func,
    options={
        'maxiter': 50,
        'verbose': True
    }
)

print(f"Optimal solution: {result['x']}")
print(f"Function value: {result['fun']}")
```

## Installation

```bash
pip install daps
```

For more installation options, see the [Installation](installation.md) page.

## When to Use DAPS

DAPS is particularly useful for:

- **Discontinuous functions**: Where gradient-based methods fail
- **Black-box optimization**: When the objective function is a "black box"
- **Scientific computing**: For complex physics-based simulations
- **Engineering design**: Where the objective function is expensive to evaluate
- **Parameter tuning**: For systems with complex parameter interactions

## Examples

<div class="grid" markdown>

<div markdown>
```python
# Basic example
from daps import daps_minimize
from daps.functions import recursive_fractal_cliff_valley_function

result = daps_minimize(recursive_fractal_cliff_valley_function)
print(f"Optimal value: {result['fun']} at {result['x']}")
```
</div>

<div markdown>
```python
# Tracking optimization progress
result = daps_minimize(
    func,
    options={
        'track_history': True,
        'verbose': True
    }
)

# Visualize the optimization path
from daps.visualization import plot_optimization_path
plot_optimization_path(result['history'], func)
```
</div>

</div>

See more in the [Examples](usage/examples.md) section.

## Visualization

DAPS includes built-in visualization tools to help understand the optimization process:

<p align="center">
<img src="assets/fig1_pas_steps_1d.png" alt="DAPS Optimization Steps" width="700">
</p>

See the [Visualizations](visualizations.md) page for more examples.

## Benchmarks

DAPS has been benchmarked against other popular optimization methods:

<p align="center">
<img src="assets/fig5_comparison_table.png" alt="Benchmark Comparison" width="700">
</p>

View detailed benchmarks on the [Benchmarks](benchmarks.md) page.

## Research

DAPS is based on research published in our paper:

> Iyer, S. (2023). Prime-Adaptive Search (PAS): A Novel Method for Efficient Optimization in Discontinuous Landscapes. [arXiv preprint arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

You can read more about the [Research Paper](paper.md) or explore the [Methodology](research/methodology.md) behind DAPS.

## Community

- [GitHub Issues](https://github.com/sethuiyer/DAPS/issues): Bug reports and feature requests
- [Discussions](https://github.com/sethuiyer/DAPS/discussions): Ask questions and share ideas
- [Contributing](contributing.md): How to contribute to DAPS development

## License

DAPS is released under the [MIT License](about/license.md). 