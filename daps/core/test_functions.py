"""
Built-in test functions for the DAPS optimizer.

This module contains various benchmark test functions commonly used
to evaluate optimization algorithms.
"""
import numpy as np
from .function import DAPSFunction

def recursive_fractal_cliff_valley_func(x: float, y: float, z: float) -> float:
    """
    Recursive Fractal Cliff Valley test function.
    
    This is a challenging 3D test function with multiple local minima,
    cliffs, and valleys. The global minimum is approximately at 
    (-π, e, √5) ≈ (-3.14159, 2.71828, 2.23607).
    
    Args:
        x: x-coordinate
        y: y-coordinate
        z: z-coordinate
        
    Returns:
        Function value at the specified point
    """
    alpha = 2.5
    beta = 0.7
    gamma = 5.0
    delta = 3.0
    
    return (np.sin(alpha * np.power(x + np.pi, 2)) + 
            np.exp(beta * np.abs(y - np.e)) +
            gamma / (1 + np.power(np.power(z - np.sqrt(5.0), 4), 1)) +
            delta * np.sin(10 * np.sin(x * y * z / 100.0)))

# Create the validated test function
recursive_fractal_cliff_valley = DAPSFunction(
    func=recursive_fractal_cliff_valley_func,
    name="Recursive Fractal Cliff Valley",
    bounds=(-15, 5, -5, 15, 0, 10),
    true_optimum=(-np.pi, np.e, np.sqrt(5)),
    true_value=None,  # Will be computed when needed
    description="A challenging 3D test function with multiple local minima, cliffs, and valleys."
)

# Compute the true value at the optimal point
recursive_fractal_cliff_valley.true_value = recursive_fractal_cliff_valley(*recursive_fractal_cliff_valley.true_optimum)

# Rosenbrock function (banana function) in 3D
def rosenbrock_3d_func(x: float, y: float, z: float) -> float:
    """
    3D Rosenbrock function.
    
    A classic test function with a narrow curved valley.
    Global minimum at (1, 1, 1) with value 0.
    
    Args:
        x, y, z: Coordinates
        
    Returns:
        Function value
    """
    return 100 * (y - x**2)**2 + (1 - x)**2 + 100 * (z - y**2)**2 + (1 - y)**2

rosenbrock_3d = DAPSFunction(
    func=rosenbrock_3d_func,
    name="Rosenbrock 3D",
    bounds=(-5, 5, -5, 5, -5, 5),
    true_optimum=(1, 1, 1),
    true_value=0.0,
    description="3D extension of the classic Rosenbrock function."
)

# Sphere function
def sphere_func(coords):
    """
    Sphere function: sum of squares of coordinates.
    
    A simple convex function with global minimum at origin (0, 0, 0) with value 0.
    
    Args:
        coords: Array of coordinates
        
    Returns:
        Function value
    """
    return np.sum(coords**2)

sphere = DAPSFunction(
    func=sphere_func,
    name="Sphere Function",
    bounds=(-10, 10, -10, 10, -10, 10),
    true_optimum=(0, 0, 0),
    true_value=0.0,
    description="Simple sphere function (sum of squares)."
)

# Ackley function in 3D
def ackley_3d_func(x: float, y: float, z: float) -> float:
    """
    Ackley function in 3D.
    
    A highly non-convex function with many local minima.
    Global minimum at (0, 0, 0) with value 0.
    
    Args:
        x, y, z: Coordinates
        
    Returns:
        Function value
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq = (x**2 + y**2 + z**2) / 3
    sum_cos = (np.cos(c*x) + np.cos(c*y) + np.cos(c*z)) / 3
    return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.exp(1)

ackley_3d = DAPSFunction(
    func=ackley_3d_func,
    name="Ackley 3D",
    bounds=(-5, 5, -5, 5, -5, 5),
    true_optimum=(0, 0, 0),
    true_value=0.0,
    description="3D version of the Ackley function, a highly non-convex test function."
)

# Rastrigin function
def rastrigin_func(coords):
    """
    Rastrigin function.
    
    A highly multimodal function with many regular local minima.
    Global minimum at origin (0, 0, 0) with value 0.
    
    Args:
        coords: Array of coordinates
        
    Returns:
        Function value
    """
    A = 10
    n = len(coords)
    return A * n + np.sum(coords**2 - A * np.cos(2 * np.pi * coords))

rastrigin = DAPSFunction(
    func=rastrigin_func,
    name="Rastrigin 3D",
    bounds=(-5.12, 5.12, -5.12, 5.12, -5.12, 5.12),
    true_optimum=(0, 0, 0),
    true_value=0.0,
    description="3D Rastrigin function, a highly multimodal test function."
) 