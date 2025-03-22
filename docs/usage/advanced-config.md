# Advanced Configuration

This page covers advanced configuration options and techniques for using DAPS in more complex scenarios.

## Optimization Parameters

DAPS offers several parameters that can be fine-tuned for specific optimization problems:

### Prime Sequence Control

```python
from daps import daps_minimize, DAPSFunction
from daps.core import set_prime_sequence, get_prime_sequence

# Get the default prime sequence
default_primes = get_prime_sequence()
print(f"Default prime sequence: {default_primes}")

# Set a custom prime sequence
set_prime_sequence([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53])

# Control which primes to use in optimization
result = daps_minimize(
    func,
    options={
        'min_prime_idx': 3,  # Start with the 3rd prime (5)
        'max_prime_idx': 10  # Don't go beyond the 10th prime (29)
    }
)
```

### Domain Shrinking Strategies

```python
# Conservative shrinking (slower but more thorough)
result = daps_minimize(
    func,
    options={
        'shrink_factor': 0.9,  # Default is 0.7
        'maxiter': 100         # May need more iterations
    }
)

# Aggressive shrinking (faster but may miss global minimum)
result = daps_minimize(
    func,
    options={
        'shrink_factor': 0.5,
        'maxiter': 30
    }
)
```

### Convergence Criteria

```python
# Tight convergence (more precise but may take longer)
result = daps_minimize(
    func,
    options={
        'tol': 1e-10,
        'maxiter': 100
    }
)

# Loose convergence (faster but less precise)
result = daps_minimize(
    func,
    options={
        'tol': 1e-4,
        'maxiter': 20
    }
)
```

## Parallel Processing

DAPS supports parallel processing for function evaluations, which can significantly speed up optimization for expensive functions:

```python
from daps import daps_minimize

# Enable parallel processing with default number of processes
result = daps_minimize(
    func,
    options={
        'parallel': True
    }
)

# Specify the number of processes
result = daps_minimize(
    func,
    options={
        'parallel': True,
        'n_jobs': 4  # Use 4 processes
    }
)
```

### Considerations for Parallel Processing

- **Function evaluation time**: Parallelization is most beneficial for functions that take significant time to evaluate
- **CPU cores**: Setting `n_jobs` higher than available cores won't provide additional speedup
- **Memory usage**: Be aware that parallelization increases memory usage
- **Thread safety**: Ensure your function is thread-safe if using parallel processing

## Custom Initial Points

You can provide an initial point to start the optimization:

```python
from daps import daps_minimize
import numpy as np

# Start optimization from a specific point
result = daps_minimize(
    func,
    x0=np.array([1.0, 2.0, 3.0])
)
```

## Handling Different Dimension Scales

When your function has dimensions with very different scales, you can use scaling strategies:

```python
from daps import daps_minimize, DAPSFunction

# Example: x is in [-1, 1], y is in [-1000, 1000], z is in [0, 0.001]
def scale_function(x, y, z):
    # Scale the inputs to similar ranges internally
    x_scaled = x
    y_scaled = y / 1000
    z_scaled = z * 1000
    
    # Your function using scaled values
    return x_scaled**2 + y_scaled**2 + z_scaled**2

func = DAPSFunction(
    func=scale_function,
    bounds=[-1, 1, -1000, 1000, 0, 0.001]
)

result = daps_minimize(func)
```

## Tracking and Analyzing Optimization

### Detailed History Tracking

```python
from daps import daps_minimize

# Enable detailed history tracking
result = daps_minimize(
    func,
    options={
        'track_history': True,
        'track_grid_points': True  # Also track all grid points (memory intensive)
    }
)

# Access the history
history = result['history']

# Analyze iterations
for i, iteration in enumerate(history):
    print(f"Iteration {i}:")
    print(f"  Prime: {iteration['prime']}")
    print(f"  Domain: {iteration['domain']}")
    print(f"  Best point: {iteration['best_point']}")
    print(f"  Function value: {iteration['best_value']}")
    
    # If track_grid_points is True
    if 'grid_points' in iteration:
        print(f"  Number of grid points: {len(iteration['grid_points'])}")
```

### Custom Callbacks

```python
from daps import daps_minimize

# Define a callback function
def my_callback(iteration_data):
    print(f"Iteration {iteration_data['iteration']}")
    print(f"Current best: {iteration_data['best_value']}")
    
    # You can analyze, save, or visualize data here
    
    # Return True to continue, False to terminate early
    return True

# Use the callback in optimization
result = daps_minimize(
    func,
    options={
        'callback': my_callback
    }
)
```

## Working with C++ Extensions

For high-performance applications, you can use the C++ interface directly:

```cpp
// In your C++ code
#include "daps/core/daps.h"

// Define your function
double my_cpp_function(double* x, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

// Run optimization
DAPSOptions options;
options.maxiter = 50;
options.min_prime_idx = 5;
options.max_prime_idx = 15;
options.tol = 1e-6;

int dim = 3;
double bounds[] = {-5.0, 5.0, -5.0, 5.0, -5.0, 5.0};

DAPSResult result = daps_minimize(my_cpp_function, dim, bounds, options);

// Process results
printf("Optimal solution: [%f, %f, %f]\n", 
       result.x[0], result.x[1], result.x[2]);
printf("Function value: %f\n", result.fun);

// Free memory
free(result.x);
```

## Optimization in Higher Dimensions

While DAPS is optimized for 3D problems, it can handle higher dimensions:

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

# Define a 4D function
def function_4d(x, y, z, w):
    return x**2 + y**2 + z**2 + w**2

# Create a DAPSFunction
func = DAPSFunction(
    func=function_4d,
    bounds=[-5, 5, -5, 5, -5, 5, -5, 5]  # [x_min, x_max, y_min, y_max, z_min, z_max, w_min, w_max]
)

# Run optimization
result = daps_minimize(func)
```

!!! warning
    Performance may degrade in higher dimensions due to the curse of dimensionality. DAPS is most efficient for 2D to 4D problems.

## Handling Constraints

DAPS works best with bound constraints, but you can handle other constraints by penalty methods:

```python
from daps import daps_minimize, DAPSFunction

# Define a function with constraints
def constrained_function(x, y, z):
    # Objective function
    f = x**2 + y**2 + z**2
    
    # Constraint: x + y + z <= 1
    constraint = x + y + z - 1
    
    # Apply penalty if constraint is violated
    if constraint > 0:
        f += 1000 * constraint**2
    
    return f

func = DAPSFunction(
    func=constrained_function,
    bounds=[-5, 5, -5, 5, -5, 5]
)

result = daps_minimize(func)
```

## Custom Error Handling

You can implement custom error handling for your function:

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

def robust_function(x, y, z):
    try:
        # Function with potential numerical issues
        result = np.log(x) + np.sqrt(y) + 1/(z+0.001)
        
        # Check for NaN or infinity
        if np.isnan(result) or np.isinf(result):
            return 1e10  # Return a large value
            
        return result
    except Exception as e:
        print(f"Warning: Function evaluation error at ({x}, {y}, {z}): {e}")
        return 1e10  # Return a large value

func = DAPSFunction(
    func=robust_function,
    bounds=[0.1, 5, 0, 5, -1, 1]
)

result = daps_minimize(func)
```

## Custom Domain Updating

Advanced users can implement custom domain updating strategies:

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

# Custom domain update function
def custom_domain_update(current_domain, best_point, iteration):
    """
    Args:
        current_domain: Current domain bounds [x_min, x_max, y_min, y_max, z_min, z_max]
        best_point: Current best point [x, y, z]
        iteration: Current iteration number
    
    Returns:
        new_domain: Updated domain bounds
    """
    # Simple example: Shrink domain around best point with adaptive factor
    adaptive_factor = 0.9 - 0.03 * min(iteration, 10)  # Shrink more aggressively over time
    
    new_domain = []
    dim = len(best_point)
    
    for i in range(dim):
        center = best_point[i]
        original_width = current_domain[2*i+1] - current_domain[2*i]
        new_width = original_width * adaptive_factor
        
        new_min = center - new_width/2
        new_max = center + new_width/2
        
        new_domain.extend([new_min, new_max])
    
    return new_domain

# Use custom domain update in optimization
result = daps_minimize(
    func,
    options={
        'custom_domain_update': custom_domain_update
    }
)
```

## Performance Considerations

### Memory Efficiency

For memory-constrained environments:

```python
result = daps_minimize(
    func,
    options={
        'track_history': False,      # Don't track history
        'track_grid_points': False,  # Don't store grid points
        'memory_efficient': True     # Use memory-efficient algorithms
    }
)
```

### Computational Efficiency

For compute-intensive functions:

```python
result = daps_minimize(
    func,
    options={
        'adaptive_sampling': True,  # Use adaptive sampling to reduce evaluations
        'parallel': True,           # Enable parallel processing
        'n_jobs': -1               # Use all available cores
    }
)
```

## Debugging and Verbose Output

For debugging purposes, you can enable different levels of verbose output:

```python
result = daps_minimize(
    func,
    options={
        'verbose': 2,  # 0: None, 1: Basic, 2: Detailed
        'debug': True  # Additional debug information
    }
)
```

## Integration with Other Libraries

### SciPy Integration

```python
from daps import daps_minimize, DAPSFunction
from scipy.optimize import minimize as scipy_minimize
import numpy as np

# Define a function
def my_function(x):
    # x is an array [x, y, z]
    return x[0]**2 + x[1]**2 + x[2]**2

# Create wrapper for DAPS
def daps_wrapper(x, y, z):
    return my_function(np.array([x, y, z]))

func = DAPSFunction(
    func=daps_wrapper,
    bounds=[-5, 5, -5, 5, -5, 5]
)

# First run DAPS
daps_result = daps_minimize(func)

# Use DAPS result as starting point for SciPy
scipy_result = scipy_minimize(
    my_function,
    x0=np.array(daps_result['x']),
    method='L-BFGS-B',
    bounds=[(-5, 5), (-5, 5), (-5, 5)]
)

print(f"DAPS result: {daps_result['fun']}")
print(f"SciPy result: {scipy_result.fun}")
```

### TensorFlow Integration

```python
import tensorflow as tf
from daps import daps_minimize, DAPSFunction

# TensorFlow function
@tf.function
def tf_function(x, y, z):
    return tf.square(x) + tf.square(y) + tf.square(z)

# Wrapper for DAPS
def daps_wrapper(x, y, z):
    # Convert to TensorFlow tensors
    x_tf = tf.constant(x, dtype=tf.float32)
    y_tf = tf.constant(y, dtype=tf.float32)
    z_tf = tf.constant(z, dtype=tf.float32)
    
    # Run TensorFlow function
    result = tf_function(x_tf, y_tf, z_tf)
    
    # Convert back to numpy/python
    return result.numpy()

func = DAPSFunction(
    func=daps_wrapper,
    bounds=[-5, 5, -5, 5, -5, 5]
)

result = daps_minimize(func)
```

## Saving and Loading Optimization State

For long-running optimizations, you can save and load the state:

```python
import pickle
from daps import daps_minimize, DAPSFunction

# Run optimization with history tracking
result = daps_minimize(
    func,
    options={
        'track_history': True,
        'maxiter': 10
    }
)

# Save state
with open('optimization_state.pkl', 'wb') as f:
    pickle.dump(result, f)

# Later, load state and continue
with open('optimization_state.pkl', 'rb') as f:
    prev_result = pickle.load(f)

# Extract information for continuing
prev_best = prev_result['x']
prev_history = prev_result['history']
prev_domain = prev_history[-1]['domain']

# Create a new DAPSFunction with updated bounds
func2 = DAPSFunction(
    func=func.func,
    name=func.name,
    bounds=prev_domain
)

# Continue optimization
result = daps_minimize(
    func2,
    x0=prev_best,
    options={
        'maxiter': 20,
        'track_history': True
    }
)

# Combine histories
full_history = prev_history + result['history']
``` 