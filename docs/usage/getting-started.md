# Getting Started with DAPS

This guide will help you quickly get started with the DAPS optimization library.

## Installation

First, install DAPS using pip:

```bash
pip install daps
```

For more installation options, refer to the [Installation](../installation.md) guide.

## Basic Concepts

DAPS (Dimensionally Adaptive Prime Search) is designed for optimizing functions, especially those with challenging characteristics like discontinuities or multiple local minima. Here are the key concepts:

- **DAPSFunction**: A wrapper for your objective function
- **daps_minimize**: The main optimization function
- **Prime-based grid sampling**: The core algorithm technique
- **Domain shrinking**: How DAPS narrows the search space

## Your First Optimization

Let's start with a simple example to find the minimum of a function:

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

# 1. Define a function to minimize
def simple_function(x, y, z):
    return x**2 + y**2 + z**2

# 2. Create a DAPSFunction instance
func = DAPSFunction(
    func=simple_function,
    name="Simple Quadratic",
    bounds=[-5, 5, -5, 5, -5, 5]  # [x_min, x_max, y_min, y_max, z_min, z_max]
)

# 3. Run the optimization
result = daps_minimize(func)

# 4. Print results
print(f"Optimal solution: {result['x']}")
print(f"Function value: {result['fun']}")
print(f"Number of function evaluations: {result['nfev']}")
```

### Understanding the Output

The `result` dictionary contains several key pieces of information:

- `result['x']`: The optimal point found (as a list [x, y, z])
- `result['fun']`: The function value at the optimal point
- `result['nfev']`: Number of function evaluations
- `result['nit']`: Number of iterations
- `result['success']`: Whether the optimization was successful
- `result['message']`: Descriptive message about termination
- `result['history']`: Optimization history (if `track_history=True`)

## Customizing the Optimization

DAPS offers several options to customize the optimization process:

```python
result = daps_minimize(
    func,
    options={
        'maxiter': 50,              # Maximum number of iterations
        'min_prime_idx': 5,         # Starting prime index
        'max_prime_idx': 15,        # Maximum prime index
        'shrink_factor': 0.7,       # Domain shrinking factor
        'tol': 1e-6,                # Convergence tolerance
        'track_history': True,      # Record optimization history
        'verbose': True             # Print progress information
    }
)
```

## Using Built-in Test Functions

DAPS includes several built-in test functions that are commonly used for benchmarking optimization algorithms:

```python
from daps import daps_minimize
from daps.functions import (
    rosenbrock_3d_function,
    ackley_3d_function,
    sphere_3d_function,
    rastrigin_3d_function,
    recursive_fractal_cliff_valley_function
)

# Minimize the Rosenbrock function
result = daps_minimize(rosenbrock_3d_function)
print(f"Rosenbrock minimum: {result['fun']} at {result['x']}")

# Try a more challenging function
result = daps_minimize(recursive_fractal_cliff_valley_function)
print(f"RFCV minimum: {result['fun']} at {result['x']}")
```

## Visualizing the Optimization

DAPS includes visualization tools to help understand the optimization process:

```python
from daps import daps_minimize, DAPSFunction
from daps.visualization import plot_optimization_path

# Define a function
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
    options={'track_history': True}
)

# Visualize the optimization path
plot_optimization_path(result['history'], func)
```

## Handling Custom Functions

DAPS can optimize any function that takes separate x, y, z arguments and returns a scalar value:

```python
def custom_function(x, y, z):
    # Your complex function here
    return np.sin(x*y) + np.cos(y*z) + x**2 + y**2 + z**2

func = DAPSFunction(
    func=custom_function,
    name="Custom Function",
    bounds=[-5, 5, -5, 5, -5, 5]
)

result = daps_minimize(func)
```

If your function takes a single array argument, you can create a wrapper:

```python
def array_function(coords):
    # coords is a numpy array [x, y, z]
    return np.sum(coords**2)

def wrapper(x, y, z):
    return array_function(np.array([x, y, z]))

func = DAPSFunction(func=wrapper, bounds=[-5, 5, -5, 5, -5, 5])
```

## Next Steps

Now that you understand the basics, you can:

1. Explore [more examples](examples.md) to see DAPS in action
2. Try the [interactive demo](interactive-demo.md) to experiment with different functions
3. Learn about [advanced configuration](advanced-config.md) options
4. Check out the [API reference](../api.md) for detailed documentation

## Troubleshooting Common Issues

### Issue: Optimization not converging

- Try increasing `maxiter`
- Adjust the `min_prime_idx` and `max_prime_idx` values
- Modify the `shrink_factor` (smaller values focus more on promising regions)

### Issue: Function evaluation errors

- Ensure your function handles all inputs within the specified bounds
- Check for division by zero or other numerical instabilities
- Use try/except blocks to handle potential errors

### Issue: Poor solution quality

- Try different starting points
- Increase the prime indices for finer grid resolution
- Consider the nature of your function (e.g., many local minima)
- Verify the bounds are appropriate for your problem 