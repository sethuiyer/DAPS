# DAPS Examples

This directory contains example scripts demonstrating how to use the DAPS (Dimensionally Adaptive Prime Search) optimization algorithm in various scenarios.

## Available Examples

### 1. Basic Example (`basic_example.py`)

A straightforward example showing how to optimize built-in and custom functions using DAPS.

```python
python basic_example.py
```

This script demonstrates:
- Basic usage of the `daps_minimize` function
- Optimizing built-in test functions
- Creating and optimizing a custom function
- Interpreting optimization results

### 2. Benchmarking (`benchmark.py`)

Compares DAPS against other optimization algorithms on various test functions.

```python
python benchmark.py
```

This script:
- Runs DAPS alongside other optimizers (Nelder-Mead, BFGS, etc.)
- Compares performance metrics (iterations, function evaluations, final error)
- Generates comparative visualizations
- Demonstrates DAPS's strengths on discontinuous functions

## Creating Your Own Examples

To create your own examples using DAPS:

1. Import the necessary modules:
   ```python
   from daps import daps_minimize, DAPSFunction
   import numpy as np
   ```

2. Define your objective function (must accept 3 parameters):
   ```python
   def my_function(x, y, z):
       return x**2 + y**2 + np.sin(z)
   ```

3. Create a DAPSFunction instance:
   ```python
   my_func = DAPSFunction(
       func=my_function,
       name="My Custom Function",
       bounds=[-5, 5, -5, 5, -5, 5],
       description="A custom 3D objective function"
   )
   ```

4. Run the optimization:
   ```python
   result = daps_minimize(
       my_func,
       options={
           'maxiter': 100,
           'min_prime_idx': 5,
           'max_prime_idx': 20,
           'tol': 1e-6
       }
   )
   ```

5. Process the results:
   ```python
   print(f"Optimal point: {result['x']}")
   print(f"Function value: {result['fun']}")
   print(f"Function evaluations: {result['nfev']}")
   ```

## Advanced Usage

For advanced usage examples, including:
- Custom callback functions
- Prime index adaptation strategies
- Handling constraints
- Integration with other optimization methods

See the documentation in the main README.md file and the API reference. 