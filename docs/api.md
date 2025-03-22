# API Reference

This page provides comprehensive documentation for the DAPS API.

## Core Functions

### daps_minimize

```python
daps_minimize(func, x0=None, options=None)
```

The primary optimization function that implements the Dimensionally Adaptive Prime Search algorithm.

**Parameters:**

- **func** (*DAPSFunction or callable*): 
  - The objective function to minimize. Can be a `DAPSFunction` instance or a callable that takes n arguments and returns a scalar value.
  
- **x0** (*array-like, optional*): 
  - Initial guess. If provided, should be an array-like object with shape matching the dimensionality of the problem.
  - If not provided, the algorithm will initialize the search at the center of the bounds.

- **options** (*dict, optional*): 
  - A dictionary of options for the optimizer:
    - **'maxiter'** (*int*): Maximum number of iterations (default: 100)
    - **'min_prime_idx'** (*int*): Starting prime index from the list of primes (default: 1)
    - **'max_prime_idx'** (*int*): Maximum prime index to use (default: 20)
    - **'shrink_factor'** (*float*): Domain shrinking factor after each iteration (default: 0.7)
    - **'tol'** (*float*): Tolerance for convergence (default: 1e-6)
    - **'track_history'** (*bool*): Whether to track optimization history (default: False)
    - **'verbose'** (*bool*): Whether to print iteration details (default: False)

**Returns:**

- **result** (*dict*): A dictionary containing:
  - **'x'** (*numpy.ndarray*): The optimal solution found.
  - **'fun'** (*float*): The function value at the optimal solution.
  - **'success'** (*bool*): Whether the optimization was successful.
  - **'nfev'** (*int*): Number of function evaluations.
  - **'nit'** (*int*): Number of iterations.
  - **'message'** (*str*): Description of the termination reason.
  - **'history'** (*list*): Optimization history if track_history=True.

**Example:**

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

# Define a 3D function to minimize
def rosenbrock(x, y, z):
    a = 1.0
    b = 100.0
    return (a - x)**2 + b * (y - x**2)**2 + (a - y)**2 + b * (z - y**2)**2

# Create a DAPSFunction instance
func = DAPSFunction(
    func=rosenbrock,
    name="Rosenbrock 3D",
    bounds=[-5, 5, -5, 5, -5, 5]
)

# Run optimization
result = daps_minimize(
    func,
    options={
        'maxiter': 50,
        'min_prime_idx': 5,
        'max_prime_idx': 15,
        'tol': 1e-6,
        'verbose': True
    }
)

print(f"Optimal solution: {result['x']}")
print(f"Function value: {result['fun']}")
print(f"Function evaluations: {result['nfev']}")
```

## Classes

### DAPSFunction

```python
DAPSFunction(func, name=None, bounds=None, dim=None)
```

A class wrapper for objective functions to be optimized using DAPS.

**Parameters:**

- **func** (*callable*): 
  - A function that takes n arguments and returns a scalar value.
  
- **name** (*str, optional*): 
  - A name for the function (for display purposes).
  
- **bounds** (*list or array-like, optional*): 
  - The bounds of the search domain. Should be specified as [x1_min, x1_max, x2_min, x2_max, ...].
  - If not provided, default bounds [-10, 10] are used for each dimension.
  
- **dim** (*int, optional*): 
  - The dimensionality of the function. If not provided, it's inferred from the bounds or the function signature.

**Methods:**

- **evaluate(x)**: 
  - Evaluates the function at point x (a list or array of coordinates).
  
- **__call__(\*args)**: 
  - Allows the function to be called directly with individual arguments.

**Example:**

```python
from daps import DAPSFunction

# Define a 2D function
def my_func(x, y):
    return x**2 + y**2

# Create a DAPSFunction
func = DAPSFunction(
    func=my_func,
    name="Simple Quadratic",
    bounds=[-5, 5, -5, 5]  # [x_min, x_max, y_min, y_max]
)

# Evaluate at a specific point
value = func(1.0, 2.0)  # Returns 5.0
# or
value = func.evaluate([1.0, 2.0])  # Returns 5.0
```

## Built-in Test Functions

DAPS includes several built-in test functions commonly used in optimization benchmarking.

### Available Test Functions

```python
from daps.functions import (
    rosenbrock_3d,
    ackley_3d,
    sphere_3d,
    rastrigin_3d,
    recursive_fractal_cliff_valley
)
```

#### Rosenbrock 3D

```python
rosenbrock_3d(x, y, z, a=1.0, b=100.0)
```

The classic banana-shaped valley test function extended to 3D.

- **Global minimum**: f(1, 1, 1) = 0

#### Ackley 3D

```python
ackley_3d(x, y, z, a=20, b=0.2, c=2*pi)
```

A highly non-convex function with many local minima.

- **Global minimum**: f(0, 0, 0) = 0

#### Sphere 3D

```python
sphere_3d(x, y, z)
```

A simple convex function defined as the sum of squares.

- **Global minimum**: f(0, 0, 0) = 0

#### Rastrigin 3D

```python
rastrigin_3d(x, y, z, A=10.0)
```

A highly multimodal function with many regular local minima.

- **Global minimum**: f(0, 0, 0) = 0

#### Recursive Fractal Cliff Valley (RFCV)

```python
recursive_fractal_cliff_valley(x, y, z)
```

A challenging function with discontinuities and fractal structure.

- **Global minimum**: approximately f(-π, e, √5) ≈ -8.14

## Visualization Tools

The `daps.visualization` module provides tools for visualizing the optimization process.

### Plot Optimization Path

```python
from daps.visualization import plot_optimization_path

plot_optimization_path(history, func, dimension_pairs=None, figsize=(12, 10), show=True)
```

Creates a visualization of the optimization path across iterations.

**Parameters:**

- **history** (*list*): The optimization history from a DAPS optimization result.
- **func** (*DAPSFunction*): The objective function.
- **dimension_pairs** (*list of tuples, optional*): Pairs of dimensions to plot (e.g., [(0,1), (1,2)]).
- **figsize** (*tuple, optional*): Figure size.
- **show** (*bool, optional*): Whether to display the plot.

### Plot Grid Evolution

```python
from daps.visualization import plot_grid_evolution

plot_grid_evolution(history, func, dims=(0, 1), steps=None, figsize=(15, 10), show=True)
```

Visualizes how the sampling grid evolves over iterations.

**Parameters:**

- **history** (*list*): The optimization history.
- **func** (*DAPSFunction*): The objective function.
- **dims** (*tuple, optional*): Which dimensions to plot.
- **steps** (*list, optional*): Which iterations to visualize.
- **figsize** (*tuple, optional*): Figure size.
- **show** (*bool, optional*): Whether to display the plot.

## Benchmark Module

The `daps.benchmark` module provides tools for benchmarking DAPS against other optimization methods.

### Run Comparison

```python
from daps.benchmark import run_comparison

run_comparison(func, methods=None, repetitions=30, options=None)
```

Compares DAPS with other optimization methods.

**Parameters:**

- **func** (*DAPSFunction*): The objective function to minimize.
- **methods** (*list, optional*): List of methods to compare (default: ["daps", "nelder-mead", "cma-es"]).
- **repetitions** (*int, optional*): Number of trials for each method.
- **options** (*dict, optional*): Options for each optimization method.

**Returns:**

- **BenchmarkResult**: An object containing the benchmark results.

## Advanced Configuration

### Custom Prime Sequences

```python
from daps.core import set_prime_sequence

set_prime_sequence([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53])
```

Sets a custom sequence of prime numbers to use for grid generation.

### Parallel Processing

```python
from daps import daps_minimize

result = daps_minimize(
    func,
    options={
        'parallel': True,
        'n_jobs': 4  # Number of parallel processes
    }
)
```

Enables parallel processing for function evaluations.

## C++ Extension Interface

For high-performance applications, DAPS provides a C++ interface that can be used to optimize C++ functions directly.

```cpp
#include "daps/core/daps.h"

// Define an objective function
double my_function(double* x, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

int main() {
    // Create optimization options
    DAPSOptions options;
    options.maxiter = 50;
    options.min_prime_idx = 5;
    options.max_prime_idx = 15;
    options.tol = 1e-6;
    
    // Define bounds
    int dim = 3;
    double bounds[] = {-5.0, 5.0, -5.0, 5.0, -5.0, 5.0};
    
    // Run optimization
    DAPSResult result = daps_minimize(my_function, dim, bounds, options);
    
    // Print results
    printf("Optimal solution: [%f, %f, %f]\n", 
           result.x[0], result.x[1], result.x[2]);
    printf("Function value: %f\n", result.fun);
    
    // Free memory
    free(result.x);
    
    return 0;
}
```

## Error Handling

DAPS provides custom exceptions for error handling:

```python
from daps.exceptions import (
    DAPSError,
    DimensionalityError,
    BoundsError,
    FunctionEvaluationError
)

try:
    result = daps_minimize(func)
except DimensionalityError as e:
    print(f"Dimensionality error: {e}")
except BoundsError as e:
    print(f"Bounds error: {e}")
except FunctionEvaluationError as e:
    print(f"Function evaluation error: {e}")
except DAPSError as e:
    print(f"General DAPS error: {e}")
``` 