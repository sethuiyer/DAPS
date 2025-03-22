"""
Unit tests for the DAPS (Dimensionally Adaptive Prime Search) package.
"""

import numpy as np
import pytest
from daps import daps_minimize, DAPSFunction
from daps.core.test_functions import (
    recursive_fractal_cliff_valley_func,
    rosenbrock_3d_func,
    sphere_func,
    ackley_func,
    rastrigin_func
)

# Test the built-in test functions
def test_built_in_functions():
    """Test that all built-in functions can be optimized."""
    built_in_funcs = [
        'recursive_fractal_cliff_valley',
        'rosenbrock_3d',
        'sphere_function',
        'ackley_function',
        'rastrigin_function'
    ]
    
    for func_name in built_in_funcs:
        result = daps_minimize(
            func_name,
            bounds=[-5, 5, -5, 5, -5, 5],
            options={'maxiter': 20, 'min_prime_idx': 5, 'max_prime_idx': 10}
        )
        
        assert isinstance(result, dict)
        assert 'x' in result
        assert 'fun' in result
        assert 'nfev' in result
        assert 'nit' in result
        assert 'success' in result
        assert 'final_prime_indices' in result
        
        assert isinstance(result['x'], np.ndarray)
        assert len(result['x']) == 3
        assert isinstance(result['fun'], float)
        assert isinstance(result['nfev'], int)
        assert isinstance(result['nit'], int)
        assert isinstance(result['success'], bool)

# Test custom function optimization
def test_custom_function():
    """Test optimization of a custom function."""
    # Define a simple custom function
    def custom_quad(x, y, z):
        return x**2 + y**2 + z**2
    
    # Run optimization
    result = daps_minimize(
        custom_quad,
        bounds=[-5, 5, -5, 5, -5, 5],
        options={'maxiter': 20}
    )
    
    # Check if result is close to the known minimum (0,0,0)
    assert np.allclose(result['x'], np.zeros(3), atol=1e-1)
    assert np.isclose(result['fun'], 0.0, atol=1e-1)

# Test DAPSFunction interface
def test_daps_function_interface():
    """Test the DAPSFunction interface."""
    # Create a DAPSFunction instance
    def custom_func(x, y, z):
        return x**2 + y**2 + z**2
    
    func = DAPSFunction(
        func=custom_func,
        name="Quadratic",
        bounds=[-5, 5, -5, 5, -5, 5],
        true_optimum=[0, 0, 0],
        true_value=0.0,
        description="Simple quadratic function"
    )
    
    # Verify attributes
    assert func.name == "Quadratic"
    assert func.bounds == [-5, 5, -5, 5, -5, 5]
    assert func.true_optimum == [0, 0, 0]
    assert func.true_value == 0.0
    assert func.description == "Simple quadratic function"
    
    # Test function evaluation
    assert func.func(1, 1, 1) == 3.0
    
    # Test optimization with the function object
    result = daps_minimize(func, options={'maxiter': 20})
    
    # Check if result is close to the known minimum
    assert np.allclose(result['x'], np.array([0, 0, 0]), atol=1e-1)
    assert np.isclose(result['fun'], 0.0, atol=1e-1)

# Test optimization options
def test_optimization_options():
    """Test different optimization options."""
    options_list = [
        {'maxiter': 10, 'min_prime_idx': 5, 'max_prime_idx': 10, 'tol': 1e-3},
        {'maxiter': 20, 'min_prime_idx': 3, 'max_prime_idx': 15, 'tol': 1e-5},
        {'maxiter': 5, 'min_prime_idx': 7, 'max_prime_idx': 12, 'tol': 1e-2},
    ]
    
    for options in options_list:
        result = daps_minimize(
            sphere_func,
            options=options
        )
        
        # Basic sanity checks
        assert result['nit'] <= options['maxiter']
        assert isinstance(result['x'], np.ndarray)
        assert len(result['x']) == 3

# Test custom callback function
def test_callback_function():
    """Test that callback function is called during optimization."""
    
    # Define a callback that counts iterations and can terminate early
    def callback(res):
        callback.count += 1
        # Stop after 5 iterations
        return callback.count < 5
    
    # Initialize counter
    callback.count = 0
    
    # Run optimization with callback
    result = daps_minimize(
        sphere_func,
        options={'maxiter': 20, 'callback': callback}
    )
    
    # Check that callback was called and stopped early
    assert callback.count == 5
    assert result['nit'] < 20  # Should have stopped before maxiter

# Test error handling
def test_error_handling():
    """Test error handling for invalid inputs."""
    
    # Test invalid function
    with pytest.raises(ValueError):
        daps_minimize('nonexistent_function')
    
    # Test invalid bounds
    with pytest.raises(ValueError):
        daps_minimize(sphere_func, bounds=[-1, 1])  # Too few bounds
    
    # Test invalid options
    with pytest.raises(TypeError):
        daps_minimize(sphere_func, options="not_a_dict")

# Test that optimization finds known minima
@pytest.mark.parametrize("func,expected_optimum,rtol", [
    (sphere_func, [0, 0, 0], 1e-1),
    (rosenbrock_3d_func, [1, 1, 1], 1e-1),
])
def test_finds_known_minima(func, expected_optimum, rtol):
    """Test that optimization finds known minima within tolerance."""
    result = daps_minimize(
        func,
        options={'maxiter': 50, 'min_prime_idx': 5, 'max_prime_idx': 15}
    )
    
    # Check if result is close to the expected optimum
    assert np.allclose(result['x'], np.array(expected_optimum), rtol=rtol)

# Run the tests if script is executed directly
if __name__ == "__main__":
    # Run all tests and capture results
    exit_code = pytest.main(["-xvs", __file__]) 