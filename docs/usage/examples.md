# Usage Examples

This page provides various examples of how to use the DAPS optimization algorithm in different scenarios.

## Basic Example

Here's a simple example of using DAPS to find the minimum of a 3D function:

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
    bounds=[-5, 5, -5, 5, -5, 5]  # [x_min, x_max, y_min, y_max, z_min, z_max]
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

## Using Built-in Test Functions

DAPS comes with several built-in test functions that you can use:

```python
from daps import daps_minimize
from daps.functions import (
    rosenbrock_3d_function,  # Returns a DAPSFunction instance
    ackley_3d_function,
    sphere_3d_function,
    rastrigin_3d_function,
    recursive_fractal_cliff_valley_function
)

# Optimize the Ackley function
result = daps_minimize(ackley_3d_function)
print(f"Ackley function optimal value: {result['fun']} at {result['x']}")

# Optimize the Recursive Fractal Cliff Valley function
result = daps_minimize(
    recursive_fractal_cliff_valley_function,
    options={'verbose': True, 'maxiter': 100}
)
print(f"RFCV function optimal value: {result['fun']} at {result['x']}")
```

## Tracking Optimization History

You can track the history of the optimization process to visualize or analyze it later:

```python
from daps import daps_minimize, DAPSFunction
from daps.visualization import plot_optimization_path

# Define your function
def my_function(x, y, z):
    return x**2 + y**2 + z**2

# Create a DAPSFunction
func = DAPSFunction(
    func=my_function,
    name="Simple Quadratic",
    bounds=[-5, 5, -5, 5, -5, 5]
)

# Run optimization with history tracking
result = daps_minimize(
    func,
    options={
        'maxiter': 20,
        'track_history': True  # Enable history tracking
    }
)

# Get the history from the result
history = result['history']
print(f"Number of iterations: {len(history)}")
print(f"Initial domain: {history[0]['domain']}")
print(f"Final domain: {history[-1]['domain']}")

# Visualize the optimization path
plot_optimization_path(history, func)
```

## Optimizing Your Own Function

Here's how to optimize a custom function with DAPS:

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

# Define a custom 3D function
def custom_3d_function(x, y, z):
    return np.sin(x*y) + np.cos(y*z) + x**2 + y**2 + z**2

# Create a DAPSFunction instance
func = DAPSFunction(
    func=custom_3d_function,
    name="Custom Function",
    bounds=[-3, 3, -3, 3, -3, 3]
)

# Run optimization
result = daps_minimize(func)

print(f"Optimal solution: {result['x']}")
print(f"Function value: {result['fun']}")
```

## Handling Discontinuous Functions

One of DAPS's strengths is handling discontinuous functions:

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

# Define a discontinuous function
def discontinuous_function(x, y, z):
    # Step function component
    step = np.floor(x) + np.floor(y) + np.floor(z)
    
    # Base function
    base = x**2 + y**2 + z**2
    
    # Combine them
    return base + 0.5 * step

# Create a DAPSFunction instance
func = DAPSFunction(
    func=discontinuous_function,
    name="Discontinuous Function",
    bounds=[-5, 5, -5, 5, -5, 5]
)

# Run optimization
result = daps_minimize(func)

print(f"Optimal solution: {result['x']}")
print(f"Function value: {result['fun']}")
```

## Advanced Configuration

DAPS offers several advanced configuration options:

```python
from daps import daps_minimize, DAPSFunction
from daps.core import set_prime_sequence

# Define your function
def my_function(x, y, z):
    return x**2 + y**2 + z**2

# Create a DAPSFunction
func = DAPSFunction(
    func=my_function,
    name="Custom Function",
    bounds=[-5, 5, -5, 5, -5, 5]
)

# Set a custom prime sequence (optional)
set_prime_sequence([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53])

# Run optimization with advanced options
result = daps_minimize(
    func,
    options={
        'maxiter': 50,              # Maximum iterations
        'min_prime_idx': 5,         # Start with the 5th prime in the sequence
        'max_prime_idx': 15,        # Don't go beyond the 15th prime
        'shrink_factor': 0.7,       # Domain shrinking factor
        'tol': 1e-8,                # Convergence tolerance
        'track_history': True,      # Track optimization history
        'verbose': True,            # Print progress
        'parallel': True,           # Enable parallel processing
        'n_jobs': 4                 # Number of parallel processes
    }
)

print(f"Optimal solution: {result['x']}")
print(f"Function value: {result['fun']}")
print(f"Function evaluations: {result['nfev']}")
```

## Comparing with Other Optimization Methods

You can compare DAPS with other optimization methods:

```python
from daps import daps_minimize, DAPSFunction
from daps.benchmark import run_comparison
import numpy as np
from scipy.optimize import minimize as scipy_minimize

# Define a function
def rosenbrock(x, y, z):
    a = 1.0
    b = 100.0
    return (a - x)**2 + b * (y - x**2)**2 + (a - y)**2 + b * (z - y**2)**2

# Create a DAPSFunction
func = DAPSFunction(
    func=rosenbrock,
    name="Rosenbrock 3D",
    bounds=[-5, 5, -5, 5, -5, 5]
)

# Run comparison
results = run_comparison(
    func,
    methods=["daps", "nelder-mead", "cma-es"],
    repetitions=10
)

# Print summary
print(results.summary())

# Plot convergence comparison
results.plot_convergence()

# Manual comparison with SciPy's minimize
def scipy_wrapper(x):
    return rosenbrock(x[0], x[1], x[2])

scipy_result = scipy_minimize(
    scipy_wrapper,
    x0=np.array([0, 0, 0]),
    method='BFGS'
)

print("\nSciPy BFGS result:")
print(f"Optimal solution: {scipy_result.x}")
print(f"Function value: {scipy_result.fun}")
print(f"Function evaluations: {scipy_result.nfev}")
```

## Visualizing the Objective Function

You can create visualizations of the objective function:

```python
from daps.visualization import plot_function_surface, plot_function_contour
from daps.functions import rosenbrock_3d_function

# Plot 3D surface
plot_function_surface(
    rosenbrock_3d_function,
    z_value=1.0,  # Fix z=1.0 to visualize a 2D slice
    x_range=(-2, 2),
    y_range=(-1, 3),
    resolution=100,
    view_angle=(30, 45)
)

# Plot contour
plot_function_contour(
    rosenbrock_3d_function,
    z_value=1.0,  # Fix z=1.0
    x_range=(-2, 2),
    y_range=(-1, 3),
    resolution=100,
    levels=20,
    show_colorbar=True
)
```

## Error Handling

DAPS provides specific error types for different situations:

```python
from daps import daps_minimize, DAPSFunction
from daps.exceptions import DAPSError, DimensionalityError, BoundsError

try:
    # This will raise a DimensionalityError
    func = DAPSFunction(
        func=lambda x, y: x + y,
        bounds=[-1, 1, -1, 1, -1, 1]  # 3D bounds for a 2D function
    )
    
    result = daps_minimize(func)
    
except DimensionalityError as e:
    print(f"Dimensionality error: {e}")
    
except BoundsError as e:
    print(f"Bounds error: {e}")
    
except DAPSError as e:
    print(f"General DAPS error: {e}")
```

## Working with NumPy Arrays

If you prefer to work with NumPy arrays directly:

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

# Define a function that takes a numpy array
def array_function(coords):
    # coords is a numpy array [x, y, z]
    return np.sum(coords**2)

# Wrapper to make it compatible with DAPSFunction
def wrapper(x, y, z):
    return array_function(np.array([x, y, z]))

# Create a DAPSFunction
func = DAPSFunction(
    func=wrapper,
    name="Array Function",
    bounds=[-5, 5, -5, 5, -5, 5]
)

# Run optimization
result = daps_minimize(func)

print(f"Optimal solution: {result['x']}")
print(f"Function value: {result['fun']}")

# Convert the result back to a numpy array if needed
optimal_coords = np.array(result['x'])
```

## Additional Resources

For more examples, check out the `examples/` directory in the GitHub repository:

- [Basic Example](https://github.com/username/daps/blob/main/examples/basic_example.py)
- [Benchmark Example](https://github.com/username/daps/blob/main/examples/benchmark.py)
- [Visualization Example](https://github.com/username/daps/blob/main/examples/visualization.py)
- [Advanced Features Example](https://github.com/username/daps/blob/main/examples/advanced_features.py) 