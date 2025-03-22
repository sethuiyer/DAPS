#!/usr/bin/env python3
"""
Simple test of the DAPS library using the pure Python implementation (base.py)
to find the value of x where x^2 = 5 by minimizing the function (x^2 - 5)^2
"""
import numpy as np
from base import DAPSFunction, daps_minimize

def target_function(x):
    """Function to minimize: (x^2 - 5)^2"""
    return (x**2 - 5)**2

if __name__ == "__main__":
    print("Testing DAPS library (pure Python implementation) by finding x such that x^2 = 5")
    print("Expected result: x ≈ ±2.236")
    
    # Multiple runs with different starting parameters
    bounds_list = [[2, 2.5], [2.2, 2.3], [0, 3]]
    best_x = None
    best_val = float('inf')
    
    for i, bounds in enumerate(bounds_list):
        print(f"\nTrial {i+1} with bounds {bounds}:")
        
        target_func = DAPSFunction(
            func=target_function,
            name="Square root of 5 finder",
            bounds=bounds,
            dimensions=1,
            description="Function to find where x^2 = 5"
        )
        
        # Optimize using DAPS with better parameters
        result = daps_minimize(
            target_func,
            options={
                'maxiter': 100,
                'min_prime_idx': 4 if i < 2 else 2,  # More points for narrower ranges
                'max_prime_idx': 12,
                'tol': 1e-10,
                'shrink_factor': 0.5,
                'improvement_factor': 0.8
            }
        )
        
        x_optimal = result['x'][0]
        fun_value = result['fun']
        
        print(f"  Optimal x: {x_optimal:.6f}")
        print(f"  Function value: {fun_value:.6e}")
        print(f"  Verification x^2: {x_optimal**2:.6f}")
        print(f"  Error: {abs(x_optimal**2 - 5):.6e}")
        print(f"  Number of iterations: {result['nit']}")
        
        if fun_value < best_val:
            best_val = fun_value
            best_x = x_optimal
    
    print("\nBest overall result:")
    print(f"Optimal x: {best_x:.6f}")
    print(f"Function value: {best_val:.6e}")
    print(f"Verification x^2: {best_x**2:.6f} (should be close to 5)")
    print(f"Error: {abs(best_x**2 - 5):.6e}")
    
    # Check if we found a good solution
    if abs(best_x**2 - 5) < 1e-4:
        print("\nSuccess! DAPS correctly found the solution.")
    else:
        print("\nWarning: The solution may not be accurate enough.")
    
    # Try a 2D example: Find the minimum of Himmelblau's function
    def himmelblau(x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    himmelblau_func = DAPSFunction(
        func=himmelblau,
        name="Himmelblau's Function",
        bounds=[-5, 5, -5, 5],  # x and y bounds
        dimensions=2,
        description="Himmelblau's function with 4 identical local minima"
    )
    
    print("\n\nSolving Himmelblau's function (a classic 2D test problem)")
    print("Expected result: One of the four minima at f(x,y) = 0:")
    print("  (3.0, 2.0), (-2.81, 3.13), (-3.78, -3.28), or (3.58, -1.85)")
    
    result_2d = daps_minimize(
        himmelblau_func,
        options={
            'maxiter': 150, 
            'min_prime_idx': 1,
            'max_prime_idx': 8
        }
    )
    
    x_opt, y_opt = result_2d['x']
    fun_val_2d = result_2d['fun']
    
    print("\nDimensionally Adaptive Prime Search (DAPS) 2D Optimization Results:")
    print(f"Optimal (x, y): ({x_opt:.4f}, {y_opt:.4f})")
    print(f"Function value: {fun_val_2d:.6e}")
    print(f"Number of iterations: {result_2d['nit']}")
    
    # Check which minimum we found (approximately)
    minima = [
        (3.0, 2.0),
        (-2.81, 3.13),
        (-3.78, -3.28),
        (3.58, -1.85)
    ]
    
    closest = min(minima, key=lambda m: np.sqrt((m[0] - x_opt)**2 + (m[1] - y_opt)**2))
    distance = np.sqrt((closest[0] - x_opt)**2 + (closest[1] - y_opt)**2)
    
    print(f"Closest known minimum: ({closest[0]:.2f}, {closest[1]:.2f})")
    print(f"Distance to that minimum: {distance:.4f}")
    
    if distance < 0.5:
        print("\nSuccess! DAPS found one of the known minima of Himmelblau's function.")
    else:
        print("\nWarning: The solution may not be accurate enough or found a different minimum.")
        
    # Try the 3D example from the README
    def custom_3d_func(x, y, z):
        return np.sin(x*y) + np.cos(y*z) + x**2 + y**2 + z**2
    
    custom_3d = DAPSFunction(
        func=custom_3d_func,
        name="Custom 3D Function",
        bounds=[-3, 3, -3, 3, -3, 3],  # 3D bounds
        dimensions=3,
        description="A custom 3D function with multiple local minima"
    )
    
    print("\n\nSolving a 3D optimization problem")
    
    result_3d = daps_minimize(
        custom_3d,
        options={
            'maxiter': 200,
            'min_prime_idx': 2,
            'max_prime_idx': 7,
            'tol': 1e-6
        }
    )
    
    x3, y3, z3 = result_3d['x']
    fun_val_3d = result_3d['fun']
    
    print("\nDimensionally Adaptive Prime Search (DAPS) 3D Optimization Results:")
    print(f"Optimal (x, y, z): ({x3:.4f}, {y3:.4f}, {z3:.4f})")
    print(f"Function value: {fun_val_3d:.6e}")
    print(f"Number of iterations: {result_3d['nit']}") 