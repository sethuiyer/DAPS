"""
Basic example demonstrating the DAPS (Dimensionally Adaptive Prime Search) algorithm.

This example shows how to:
1. Use the built-in test functions
2. Define a custom function
3. Visualize the optimization process
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from daps import daps_minimize, DAPSFunction
import time

# Example 1: Optimize a built-in test function
print("Example 1: Optimizing built-in Recursive Fractal Cliff Valley function")
start_time = time.time()
result = daps_minimize(
    'recursive_fractal_cliff_valley',
    bounds=[-5, 5, -5, 5, -5, 5],
    options={
        'maxiter': 500,
        'min_prime_idx': 5,
        'max_prime_idx': 15
    }
)
end_time = time.time()

# Print the results
print(f"Optimization completed in {end_time - start_time:.2f} seconds")
print(f"Best solution: x = {result['x']}")
print(f"Function value: {result['fun']}")
print(f"Number of function evaluations: {result['nfev']}")
print(f"Number of iterations: {result['nit']}")
print(f"Success: {result['success']}")
print(f"Final prime indices: {result['final_prime_indices']}")
print()

# Example 2: Define and optimize a custom function
print("Example 2: Optimizing a custom function")

# Define a custom function
def custom_function(x, y, z):
    """A custom 3D function with multiple local minima"""
    return np.sin(x*y) + np.cos(y*z) + x**2 + 0.5*y**2 + 0.25*z**2

# Create a DAPSFunction instance with metadata
custom_daps_func = DAPSFunction(
    func=custom_function,
    name="Custom Sine-Cosine Quadratic",
    bounds=[-5, 5, -5, 5, -5, 5],
    true_optimum=None,  # We don't know the true optimum
    true_value=None,    # We don't know the true value
    description="A custom 3D function with multiple local minima combining sinusoidal and quadratic terms"
)

# Define progress callback
def progress_callback(result):
    """Callback function to monitor optimization progress"""
    iter_num = progress_callback.count
    if iter_num % 5 == 0:  # Print every 5 iterations
        print(f"Iteration {iter_num}: Best value = {result['fun']:.6f} at {result['x']}")
    progress_callback.count += 1
    return True  # Continue optimization

# Initialize callback counter
progress_callback.count = 0

# Run optimization with callback
start_time = time.time()
result = daps_minimize(
    custom_daps_func,
    options={
        'maxiter': 100,
        'callback': progress_callback,
        'tol': 1e-10
    }
)
end_time = time.time()

# Print the results
print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
print(f"Best solution: x = {result['x']}")
print(f"Function value: {result['fun']}")
print(f"Number of function evaluations: {result['nfev']}")
print(f"Number of iterations: {result['nit']}")
print(f"Success: {result['success']}")
print()

# Example 3: Visualization of the optimization landscape
print("Example 3: Visualizing the optimization landscape")

# Create a grid of points
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Fix z at the optimal value for visualization
optimal_z = result['x'][2]

# Compute function values
for i in range(len(x)):
    for j in range(len(y)):
        Z[j, i] = custom_function(X[j, i], Y[j, i], optimal_z)

# Create 3D surface plot
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Function Value')
ax1.set_title('3D Surface of Custom Function')

# Add the optimization result point
ax1.scatter(
    result['x'][0], result['x'][1], result['fun'],
    color='red', s=100, marker='*', label='Optimum'
)
ax1.legend()

# Create contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title(f'Contour with Z fixed at {optimal_z:.4f}')

# Add the optimization result point
ax2.scatter(
    result['x'][0], result['x'][1],
    color='red', s=100, marker='*', label='Optimum'
)
ax2.legend()

plt.colorbar(contour, ax=ax2, label='Function Value')
plt.tight_layout()
plt.savefig('optimization_landscape.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'optimization_landscape.png'") 