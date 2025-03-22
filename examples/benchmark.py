#!/usr/bin/env python3
"""
Benchmark script for the DAPS optimization algorithm.
Uses pytest-benchmark to measure performance against other optimizers.

Run with:
    pytest benchmark.py --benchmark-json=benchmark_results.json
    
For GitHub Actions integration, this script is automatically run
and results are published to the project's GitHub Pages.
"""
import numpy as np
import pytest
from scipy import optimize

from daps import DAPSFunction, daps_minimize
from daps.core.test_functions import (
    recursive_fractal_cliff_valley,
    rosenbrock_3d,
    sphere_function,
    ackley_3d,
    rastrigin_function
)

# Dictionary of test functions
TEST_FUNCTIONS = {
    "Recursive Fractal Cliff Valley": recursive_fractal_cliff_valley,
    "Rosenbrock 3D": rosenbrock_3d,
    "Sphere": sphere_function,
    "Ackley 3D": ackley_3d,
    "Rastrigin": rastrigin_function
}

# Dictionary to store the number of function evaluations
func_evals = {name: 0 for name in TEST_FUNCTIONS}

def reset_func_evals():
    """Reset function evaluation counters."""
    for key in func_evals:
        func_evals[key] = 0

def function_wrapper(func_name):
    """Wraps a test function to count evaluations."""
    def wrapped(x, y, z):
        func_evals[func_name] += 1
        return TEST_FUNCTIONS[func_name](x, y, z)
    return wrapped

def get_bounds(func_name):
    """Get bounds for each test function."""
    if func_name == "Recursive Fractal Cliff Valley":
        return [-10, 10, -10, 10, -10, 10]
    elif func_name == "Rosenbrock 3D":
        return [-5, 5, -5, 5, -5, 5]
    elif func_name == "Sphere":
        return [-10, 10, -10, 10, -10, 10]
    elif func_name == "Ackley 3D":
        return [-32.768, 32.768, -32.768, 32.768, -32.768, 32.768]
    elif func_name == "Rastrigin":
        return [-5.12, 5.12, -5.12, 5.12, -5.12, 5.12]
    return [-10, 10, -10, 10, -10, 10]

# Benchmark DAPS against other optimizers
@pytest.mark.parametrize("func_name", list(TEST_FUNCTIONS.keys()))
def test_daps_benchmark(benchmark, func_name):
    """Benchmark DAPS optimizer."""
    reset_func_evals()
    wrapped_func = function_wrapper(func_name)
    bounds = get_bounds(func_name)
    
    daps_func = DAPSFunction(
        func=wrapped_func,
        name=func_name,
        bounds=bounds
    )
    
    # Using benchmark to measure performance
    result = benchmark(
        daps_minimize,
        daps_func,
        options={
            'maxiter': 30,
            'min_prime_idx': 5,
            'max_prime_idx': 20
        }
    )
    
    # Test that the optimization worked
    assert result['success']
    
    # Save function evaluations for comparison
    function_evals = func_evals[func_name]
    
    return {
        'optimizer': 'DAPS',
        'function': func_name,
        'value': result['fun'],
        'evaluations': function_evals,
        'iterations': result['nit']
    }

# For comparison: Nelder-Mead
@pytest.mark.parametrize("func_name", list(TEST_FUNCTIONS.keys()))
def test_nelder_mead_benchmark(benchmark, func_name):
    """Benchmark Nelder-Mead optimizer for comparison."""
    reset_func_evals()
    wrapped_func = function_wrapper(func_name)
    bounds = get_bounds(func_name)
    
    def scipy_func(params):
        x, y, z = params
        return wrapped_func(x, y, z)
    
    x0 = np.random.uniform(
        [bounds[0], bounds[2], bounds[4]],
        [bounds[1], bounds[3], bounds[5]]
    )
    
    # Using benchmark to measure performance
    result = benchmark(
        optimize.minimize,
        scipy_func,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 200}
    )
    
    # Save function evaluations for comparison
    function_evals = func_evals[func_name]
    
    return {
        'optimizer': 'Nelder-Mead',
        'function': func_name,
        'value': result.fun,
        'evaluations': function_evals,
        'iterations': result.nit,
        'success': result.success
    }

# For comparison: L-BFGS-B (for smooth functions only)
@pytest.mark.parametrize("func_name", ["Sphere", "Rosenbrock 3D"])
def test_lbfgs_benchmark(benchmark, func_name):
    """Benchmark L-BFGS-B optimizer for comparison (smooth functions only)."""
    reset_func_evals()
    wrapped_func = function_wrapper(func_name)
    bounds_list = get_bounds(func_name)
    
    def scipy_func(params):
        x, y, z = params
        return wrapped_func(x, y, z)
    
    # Convert bounds to scipy format
    scipy_bounds = [
        (bounds_list[0], bounds_list[1]),
        (bounds_list[2], bounds_list[3]),
        (bounds_list[4], bounds_list[5])
    ]
    
    x0 = np.random.uniform(
        [bounds_list[0], bounds_list[2], bounds_list[4]],
        [bounds_list[1], bounds_list[3], bounds_list[5]]
    )
    
    # Using benchmark to measure performance
    result = benchmark(
        optimize.minimize,
        scipy_func,
        x0,
        method='L-BFGS-B',
        bounds=scipy_bounds,
        options={'maxiter': 100}
    )
    
    # Save function evaluations for comparison
    function_evals = func_evals[func_name]
    
    return {
        'optimizer': 'L-BFGS-B',
        'function': func_name,
        'value': result.fun,
        'evaluations': function_evals,
        'iterations': result.nit,
        'success': result.success
    }

if __name__ == "__main__":
    print("This script is designed to be run with pytest-benchmark:")
    print("pytest benchmark.py --benchmark-json=benchmark_results.json") 