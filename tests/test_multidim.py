"""
Unit tests for multi-dimensional optimization capabilities of DAPS.
"""

import numpy as np
import pytest
from daps import daps_minimize, DAPSFunction

# 1D test functions
def parabola_1d(x):
    """Simple 1D parabola function with minimum at x=2"""
    return (x - 2) ** 2

def sinusoidal_1d(x):
    """Sinusoidal function with multiple local minima"""
    return np.sin(x) + 0.1 * x

# 2D test functions
def rosenbrock_2d(x, y):
    """Rosenbrock function in 2D"""
    return 100 * (y - x**2)**2 + (1 - x)**2

def himmelblau_2d(x, y):
    """Himmelblau function with multiple local minima"""
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# 3D test function for comparison
def paraboloid_3d(x, y, z):
    """Simple 3D paraboloid with minimum at (1, 2, 3)"""
    return (x - 1)**2 + (y - 2)**2 + (z - 3)**2

# Test 1D optimization
def test_1d_optimization():
    """Test optimization of 1D functions"""
    # Test parabola function
    result = daps_minimize(
        parabola_1d,
        bounds=[-10, 10],
        options={'dimensions': 1, 'max_iterations': 50}
    )
    
    # Verify results
    assert len(result['x']) == 1
    assert np.isclose(result['x'][0], 2.0, atol=0.1)
    assert np.isclose(result['fun'], 0.0, atol=0.01)
    assert 'dimensions' in result
    assert result['dimensions'] == 1
    assert 'final_prime_idx_x' in result
    
    # Auto-detect dimensions from bounds
    result2 = daps_minimize(
        parabola_1d,
        bounds=[-10, 10],  # 1D bounds
        options={'max_iterations': 50}
    )
    assert result2['dimensions'] == 1
    assert len(result2['x']) == 1
    
    # Test with DAPSFunction
    func = DAPSFunction(
        func=parabola_1d,
        name="Parabola 1D",
        bounds=[-10, 10],
        dimensions=1,
        true_optimum=[2.0],
        true_value=0.0
    )
    
    result3 = daps_minimize(func, options={'max_iterations': 50})
    assert result3['dimensions'] == 1
    assert np.isclose(result3['x'][0], 2.0, atol=0.1)

# Test 2D optimization
def test_2d_optimization():
    """Test optimization of 2D functions"""
    # Test Rosenbrock function
    result = daps_minimize(
        rosenbrock_2d,
        bounds=[-5, 5, -5, 5],
        options={'dimensions': 2, 'max_iterations': 70}
    )
    
    # Verify results
    assert len(result['x']) == 2
    assert np.allclose(result['x'], [1.0, 1.0], atol=0.2)
    assert np.isclose(result['fun'], 0.0, atol=0.1)
    assert result['dimensions'] == 2
    assert 'final_prime_idx_x' in result
    assert 'final_prime_idx_y' in result
    assert 'final_prime_idx_z' not in result
    
    # Auto-detect dimensions from bounds
    result2 = daps_minimize(
        rosenbrock_2d,
        bounds=[-5, 5, -5, 5],  # 2D bounds
        options={'max_iterations': 70}
    )
    assert result2['dimensions'] == 2
    assert len(result2['x']) == 2
    
    # Test with DAPSFunction
    func = DAPSFunction(
        func=rosenbrock_2d,
        name="Rosenbrock 2D",
        bounds=[-5, 5, -5, 5],
        dimensions=2,
        true_optimum=[1.0, 1.0],
        true_value=0.0
    )
    
    result3 = daps_minimize(func, options={'max_iterations': 70})
    assert result3['dimensions'] == 2
    assert np.allclose(result3['x'], [1.0, 1.0], atol=0.2)

# Test 3D optimization
def test_3d_optimization():
    """Test optimization of 3D functions"""
    # Test paraboloid function
    result = daps_minimize(
        paraboloid_3d,
        bounds=[-10, 10, -10, 10, -10, 10],
        options={'dimensions': 3, 'max_iterations': 50}
    )
    
    # Verify results
    assert len(result['x']) == 3
    assert np.allclose(result['x'], [1.0, 2.0, 3.0], atol=0.2)
    assert np.isclose(result['fun'], 0.0, atol=0.01)
    assert result['dimensions'] == 3
    assert 'final_prime_idx_x' in result
    assert 'final_prime_idx_y' in result
    assert 'final_prime_idx_z' in result
    
    # Auto-detect dimensions from bounds
    result2 = daps_minimize(
        paraboloid_3d,
        bounds=[-10, 10, -10, 10, -10, 10],  # 3D bounds
        options={'max_iterations': 50}
    )
    assert result2['dimensions'] == 3
    assert len(result2['x']) == 3

# Test array-input functions
def test_array_input_functions():
    """Test functions that take array inputs instead of individual parameters"""
    # 1D array input function
    def array_func_1d(x):
        if isinstance(x, np.ndarray):
            return (x[0] - 2) ** 2
        else:
            return (x - 2) ** 2
    
    result1 = daps_minimize(
        array_func_1d,
        bounds=[-10, 10],
        options={'dimensions': 1}
    )
    assert np.isclose(result1['x'][0], 2.0, atol=0.1)
    
    # 2D array input function
    def array_func_2d(x):
        if isinstance(x, np.ndarray):
            return (x[0] - 1)**2 + (x[1] - 2)**2
        else:
            return (x - 1)**2 + (y - 2)**2
    
    result2 = daps_minimize(
        array_func_2d,
        bounds=[-10, 10, -10, 10],
        options={'dimensions': 2}
    )
    assert np.allclose(result2['x'], [1.0, 2.0], atol=0.1)
    
    # 3D array input function
    def array_func_3d(x):
        if isinstance(x, np.ndarray):
            return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2
        else:
            return (x - 1)**2 + (y - 2)**2 + (z - 3)**2
    
    result3 = daps_minimize(
        array_func_3d,
        bounds=[-10, 10, -10, 10, -10, 10],
        options={'dimensions': 3}
    )
    assert np.allclose(result3['x'], [1.0, 2.0, 3.0], atol=0.1)

# Test error cases
def test_error_cases():
    """Test error handling for dimension-related issues"""
    # Mismatch between dimensions and bounds
    with pytest.raises(ValueError):
        daps_minimize(
            parabola_1d,
            bounds=[-10, 10],
            options={'dimensions': 2}
        )
    
    # Invalid dimension value
    with pytest.raises(ValueError):
        daps_minimize(
            parabola_1d,
            bounds=[-10, 10],
            options={'dimensions': 4}
        )
    
    # Invalid function for the given dimensions
    def only_3d_func(x, y, z):
        return x**2 + y**2 + z**2
    
    with pytest.raises(ValueError):
        daps_minimize(
            only_3d_func,
            bounds=[-10, 10],
            options={'dimensions': 1}
        )

# Test dimension conversion for built-in functions
def test_built_in_functions_dimensions():
    """Test that built-in functions can be called with different dimensions"""
    # Test with sphere function in 1D
    result1 = daps_minimize(
        'sphere_function',
        bounds=[-10, 10],
        options={'dimensions': 1}
    )
    assert result1['dimensions'] == 1
    assert len(result1['x']) == 1
    assert np.isclose(result1['x'][0], 0.0, atol=0.1)
    
    # Test with sphere function in 2D
    result2 = daps_minimize(
        'sphere_function',
        bounds=[-10, 10, -10, 10],
        options={'dimensions': 2}
    )
    assert result2['dimensions'] == 2
    assert len(result2['x']) == 2
    assert np.allclose(result2['x'], [0.0, 0.0], atol=0.1)
    
    # Test with Rastrigin function in 2D
    result3 = daps_minimize(
        'rastrigin_function',
        bounds=[-5.12, 5.12, -5.12, 5.12],
        options={'dimensions': 2, 'max_iterations': 100}
    )
    assert result3['dimensions'] == 2
    assert len(result3['x']) == 2
    assert np.allclose(result3['x'], [0.0, 0.0], atol=0.1)

if __name__ == "__main__":
    # Run all tests and capture results
    exit_code = pytest.main(["-xvs", __file__]) 