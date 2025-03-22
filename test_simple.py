#!/usr/bin/env python3
"""
Simple test of the DAPS library to find the value of x where x^2 = 5
by minimizing the function (x^2 - 5)^2
"""
from daps import daps_minimize
import numpy as np

def target_function(x):
    """Function to minimize: (x^2 - 5)^2"""
    return (x**2 - 5)**2

if __name__ == "__main__":
    print("Testing DAPS library by finding x such that x^2 = 5")
    print("Expected result: x ≈ ±2.236")
    
    # Find the positive solution
    result = daps_minimize(
        target_function,
        bounds=[0, 5],  # Search in the positive range
        options={
            'dimensions': 1,
            'maxiter': 100,
            'verbose': True
        }
    )
    
    x_optimal = result['x'][0]
    fun_value = result['fun']
    
    print("\nDoctors Appointment Scheduling (DAPS) Optimization Results:")
    print(f"Optimal x: {x_optimal:.6f}")
    print(f"Function value: {fun_value:.6e}")
    print(f"Verification x^2: {x_optimal**2:.6f} (should be close to 5)")
    print(f"Error: {abs(x_optimal**2 - 5):.6e}")
    print(f"Number of function evaluations: {result['nfev']}")
    
    # Check if we found a good solution
    if abs(x_optimal**2 - 5) < 1e-4:
        print("\nSuccess! DAPS correctly found the solution.")
    else:
        print("\nWarning: The solution may not be accurate enough.") 