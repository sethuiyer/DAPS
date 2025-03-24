# DAPS - Dimensionally Adaptive Prime Search

<img src="./LOGO.webp" height="256px" width="256px" alt="DAPS Logo" align="right"/>

A high-performance global optimization algorithm for 1D, 2D, and 3D functions, implemented in C++ with Python bindings.

## How It Works

DAPS uses prime number-based grid sampling to avoid aliasing problems common in regular grid search methods. It dynamically adapts resolution and shrinks the search domain around promising regions.
It assumes a measurable loss function at every evaluation. Primes are treated as resolution knobs and can be increased or decreased depending on the degree of accuracy needed.

For theoretical details, see the [research paper](paper/build/daps_paper.pdf).


## 🎧 DAPS Podcast Episode

Listen to the introduction of **Dimensionally Adaptive Prime Search (DAPS)** — the story, the math, and the future: [Podcast](https://github.com/sethuiyer/DAPS/blob/main/daps_podcast.mp3)

[Hacker News Thread](https://news.ycombinator.com/item?id=43451114)

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

Here’s a **PyTorch‑compatible DAPS optimizer** :

- Starts at prime=97  
- Never drops below prime=2  
- Works for **n‑dimensional** functions in batch (GPU‑ready)  
- Adapts prime resolution, shrinks domain, and clamps to your original bounds  

```python
import torch
from sympy import primerange

class DAPSOptimizerTorch:
    def __init__(self, bounds, device='cpu', prime_start=97):
        primes = list(primerange(2,500))
        self.prime_list = primes
        self.prime_idx = primes.index(prime_start)
        self.min_idx = 0
        self.max_idx = len(primes)-1
        self.device = torch.device(device)
        self.bounds = torch.tensor(bounds, device=self.device).view(-1,2)

    def optimize(self, func, maxiter=10, samples=1000, shrink=0.5, tol=1e-6):
        domain = self.bounds.clone()
        best_val, best_x = float('inf'), None

        for _ in range(maxiter):
            p = self.prime_list[self.prime_idx]
            pts = domain[:,0] + (domain[:,1]-domain[:,0]) * torch.rand(samples, self.bounds.size(0), device=self.device)
            vals = func(pts).flatten()
            idx = torch.argmin(vals)
            val, x = vals[idx].item(), pts[idx]

            if val < best_val:
                best_val, best_x = val, x.clone()
                self.prime_idx = min(self.prime_idx+1, self.max_idx)
            else:
                self.prime_idx = max(self.prime_idx-1, self.min_idx)

            span = domain[:,1] - domain[:,0]
            domain[:,0] = torch.max(self.bounds[:,0], best_x - span*shrink/2)
            domain[:,1] = torch.min(self.bounds[:,1], best_x + span*shrink/2)

            if best_val < tol:
                break

        return best_x.cpu().numpy(), best_val
```
That’s your **n‑dimensional, GPU‑ready, prime‑adaptive optimizer**.

### 🔥 Another Example

```python
import numpy as np
from sympy import primerange
import matplotlib.pyplot as plt

# Define the DAPSFunction container
class DAPSFunction:
    def __init__(self, func, name, bounds, dimensions,
                 true_optimum=None, true_value=None, description=""):
        self.func = func
        self.name = name
        self.bounds = bounds
        self.dimensions = dimensions
        self.true_optimum = true_optimum
        self.true_value = true_value
        self.description = description

# DAPS algorithm
def daps_minimize(daps_func, options=None):
    if not isinstance(daps_func, DAPSFunction):
        raise ValueError("daps_minimize expects a DAPSFunction object.")

    if options is None:
        options = {}
    maxiter = options.get('maxiter', 100)
    min_prime_idx = options.get('min_prime_idx', 0)
    max_prime_idx = options.get('max_prime_idx', 10)
    tol = options.get('tol', 1e-6)
    alpha = options.get('shrink_factor', 0.5)
    improvement_factor = options.get('improvement_factor', 0.9)

    prime_list = list(primerange(2, 2000))
    prime_index = min_prime_idx

    dims = daps_func.dimensions
    domain_array = np.array(daps_func.bounds).reshape(dims, 2)

    best_x, best_val = None, np.inf
    trace = []

    for iteration in range(1, maxiter + 1):
        p = prime_list[prime_index]
        grid_axes = [np.linspace(domain_array[d, 0], domain_array[d, 1], p) for d in range(dims)]
        mesh = np.meshgrid(*grid_axes)
        samples = np.vstack([m.ravel() for m in mesh]).T

        if dims == 1:
            evals = np.array([daps_func.func(xi[0]) for xi in samples])
        else:
            evals = np.array([daps_func.func(*pt) for pt in samples])

        idx_best = np.argmin(evals)
        local_best_x = samples[idx_best]
        local_best_val = evals[idx_best]

        if local_best_val < best_val:
            ratio_improvement = 0 if best_val == np.inf else (best_val - local_best_val) / abs(best_val)
            best_val = local_best_val
            best_x = local_best_x

        if best_val < tol:
            break

        if best_val < np.inf and ratio_improvement > (1 - improvement_factor):
            prime_index = min(prime_index + 1, max_prime_idx)
        else:
            prime_index = max(prime_index - 1, min_prime_idx)

        for d in range(dims):
            current_span = domain_array[d, 1] - domain_array[d, 0]
            domain_array[d, 0] = best_x[d] - alpha * current_span / 2
            domain_array[d, 1] = best_x[d] + alpha * current_span / 2

        trace.append(best_val)

    return {
        'x': best_x,
        'fun': best_val,
        'nit': iteration,
        'trace': trace
    }

# Rastrigin function setup
def rastrigin_2d(x, y):
    A = 10
    return A * 2 + (x ** 2 - A * np.cos(2 * np.pi * x)) + (y ** 2 - A * np.cos(2 * np.pi * y))

rastrigin = DAPSFunction(
    func=rastrigin_2d,
    name="Rastrigin 2D",
    bounds=[-5.12, 5.12, -5.12, 5.12],
    dimensions=2
)

# Run DAPS on Rastrigin
result = daps_minimize(rastrigin, {'maxiter': 100, 'max_prime_idx': 12})

# Plot convergence
plt.figure(figsize=(8, 5))
plt.plot(result['trace'], marker='o')
plt.title("DAPS Convergence on Rastrigin 2D")
plt.xlabel("Iteration")
plt.ylabel("Best Function Value")
plt.grid(True)
plt.tight_layout()

result, plt.show()

```
![image](https://github.com/user-attachments/assets/3374b272-1ca1-44eb-b7c6-7a489a44c712)

```text
2.4715367774273638
2.4715367774273638
1.993388867879645
1.993388867879645
1.9915709523838139
1.9905800313279514
1.9901169338024438
1.9901169338024438
1.9900346001631561
1.9900346001631561
1.9899208481938544
1.9899208481938544
1.9899202403985718
1.9899181237242267
```

### 💥 **Observation**:
- **Iteration ~4:** Big drop from **27 → 2.47**
- **Iteration ~6-10:** Creeps down toward 1.99
- **Iteration ~17-20:** Hits **1.9899181237242267**
- **Iteration ~21:** **1.9899181141865796** — stays constant **forever** after this.

### ✅ **Convergence Point: ~21 iterations**
**After iteration 21**, no further improvement 


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
