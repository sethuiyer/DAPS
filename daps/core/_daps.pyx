# _daps.pyx
# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.functional cimport function
from libcpp.tuple cimport tuple as cpp_tuple
import cython
from cpython.ref cimport PyObject

# Python callback registry to avoid garbage collection
cdef dict _callback_registry = {}

# Define C++ functions from daps.cpp
cdef extern from "daps.cpp":
    # Built-in test functions
    double recursive_fractal_cliff_valley(double x, double y, double z) nogil
    double rosenbrock_3d(double x, double y, double z) nogil
    double sphere_function(double x, double y, double z) nogil
    double ackley_function(double x, double y, double z) nogil
    double rastrigin_function(double x, double y, double z) nogil
    
    # DAPS algorithm
    cpp_tuple[vector[double], double, int, int, bool, int, int] daps_optimize(
        function[double(double, double, double)] func,
        double x_min, double x_max,
        double y_min, double y_max,
        double z_min, double z_max,
        int max_iter,
        int min_prime_idx,
        int max_prime_idx,
        function[bool(vector[double], double, int)] callback,
        double tol
    ) nogil

# Type definitions for function pointers
ctypedef double (*c_func_type)(double, double, double) nogil
ctypedef bool (*callback_func_type)(vector[double], double, int) nogil

# Wrapper for callback functions from Python to C++
cdef bool python_callback_wrapper(vector[double]& x, double fun_val, int evals, object py_callback) with gil:
    """Wrapper to call Python callback from C++"""
    try:
        # Call Python callback with result dict
        result = {
            'x': np.array([x[0], x[1], x[2]]),
            'fun': fun_val,
            'nfev': evals
        }
        return py_callback(result)
    except:
        # On error, return False to stop optimization
        return False

# Factory for creating callback wrapper
cdef callback_func_type create_callback_wrapper(object py_callback):
    """Create a C++ callback function that will call the Python callback"""
    if py_callback is None:
        return NULL
    
    # Store Python callback in registry
    global _callback_registry
    cdef int callback_id = id(py_callback)
    _callback_registry[callback_id] = py_callback
    
    # Create wrapper that captures the callback_id
    cdef callback_func_type callback_wrapper = (
        lambda vector[double] x, double fun_val, int evals:
        python_callback_wrapper(x, fun_val, evals, _callback_registry[callback_id])
    )
    
    return callback_wrapper

# Wrapper for user-defined Python functions
cdef double python_function_wrapper(double x, double y, double z, object py_func) with gil:
    """Wrapper to call Python function from C++"""
    try:
        return py_func(x, y, z)
    except:
        # On error, return a large value
        return 1e100

# Factory for creating function wrapper
cdef function[double(double, double, double)] create_function_wrapper(object py_func):
    """Create a C++ function that will call the Python function"""
    # Store the Python function in registry to prevent garbage collection
    global _callback_registry
    cdef int func_id = id(py_func)
    _callback_registry[func_id] = py_func
    
    # Create wrapper that captures the func_id
    cdef function[double(double, double, double)] func_wrapper = (
        lambda double x, double y, double z:
        python_function_wrapper(x, y, z, _callback_registry[func_id])
    )
    
    return func_wrapper

# Python-accessible functions for built-in test functions
def py_recursive_fractal_cliff_valley(double x, double y, double z):
    """Python wrapper for the Recursive Fractal Cliff Valley test function"""
    return recursive_fractal_cliff_valley(x, y, z)

def py_rosenbrock_3d(double x, double y, double z):
    """Python wrapper for the Rosenbrock 3D test function"""
    return rosenbrock_3d(x, y, z)

def py_sphere_function(double x, double y, double z):
    """Python wrapper for the Sphere test function"""
    return sphere_function(x, y, z)

def py_ackley_function(double x, double y, double z):
    """Python wrapper for the Ackley test function"""
    return ackley_function(x, y, z)

def py_rastrigin_function(double x, double y, double z):
    """Python wrapper for the Rastrigin test function"""
    return rastrigin_function(x, y, z)

# Main Python-accessible DAPS function
def daps_cython(object func, list bounds, dict options=None):
    """
    Cython implementation of the DAPS algorithm.
    
    Parameters
    ----------
    func : callable or built-in function
        The objective function to be minimized
    bounds : list or tuple
        The bounds for each variable [x_min, x_max, y_min, y_max, z_min, z_max]
    options : dict, optional
        Options for the optimizer:
        - maxiter: Maximum number of iterations (default: 1000)
        - min_prime_idx: Minimum prime index (default: 5)
        - max_prime_idx: Maximum prime index (default: 20)
        - callback: Callback function called after each iteration (default: None)
        - tol: Tolerance for termination (default: 1e-8)
    
    Returns
    -------
    dict
        The optimization results containing:
        - 'x': The optimal solution
        - 'fun': The function value at the optimum
        - 'nfev': Number of function evaluations
        - 'nit': Number of iterations
        - 'success': Whether the optimization was successful
        - 'final_prime_indices': Final prime indices used
    """
    # Default options
    cdef dict default_options = {
        'maxiter': 1000,
        'min_prime_idx': 5,
        'max_prime_idx': 20,
        'callback': None,
        'tol': 1e-8
    }
    
    # Merge options
    if options is None:
        options = {}
    
    cdef dict merged_options = default_options.copy()
    merged_options.update(options)
    
    # Extract options
    cdef int maxiter = merged_options['maxiter']
    cdef int min_prime_idx = merged_options['min_prime_idx']
    cdef int max_prime_idx = merged_options['max_prime_idx']
    cdef object py_callback = merged_options['callback']
    cdef double tol = merged_options['tol']
    
    # Validate bounds
    if len(bounds) != 6:
        raise ValueError("Bounds must contain 6 values [x_min, x_max, y_min, y_max, z_min, z_max]")
    
    cdef double x_min = bounds[0]
    cdef double x_max = bounds[1]
    cdef double y_min = bounds[2]
    cdef double y_max = bounds[3]
    cdef double z_min = bounds[4]
    cdef double z_max = bounds[5]
    
    # Create callback wrapper if provided
    cdef callback_func_type callback_wrapper = create_callback_wrapper(py_callback)
    
    # Check if using built-in function or Python function
    cdef function[double(double, double, double)] func_wrapper
    
    # Determine the function to use
    if hasattr(func, '__call__'):
        # Python function - needs wrapping
        func_wrapper = create_function_wrapper(func)
    elif func == 'recursive_fractal_cliff_valley':
        func_wrapper = recursive_fractal_cliff_valley
    elif func == 'rosenbrock_3d':
        func_wrapper = rosenbrock_3d
    elif func == 'sphere_function':
        func_wrapper = sphere_function
    elif func == 'ackley_function':
        func_wrapper = ackley_function
    elif func == 'rastrigin_function':
        func_wrapper = rastrigin_function
    else:
        raise ValueError(f"Unknown built-in function: {func}")
    
    # Run the optimization (release GIL for better parallelism)
    cdef cpp_tuple[vector[double], double, int, int, bool, int, int] result
    with nogil:
        result = daps_optimize(
            func_wrapper,
            x_min, x_max,
            y_min, y_max,
            z_min, z_max,
            maxiter,
            min_prime_idx,
            max_prime_idx,
            callback_wrapper,
            tol
        )
    
    # Extract results
    cdef vector[double] x_best = result[0]
    cdef double fun_best = result[1]
    cdef int nfev = result[2]
    cdef int nit = result[3]
    cdef bool success = result[4]
    cdef int final_p_idx_x = result[5]
    cdef int final_p_idx_y = result[6]
    
    # Clean up the callback registry
    if py_callback is not None:
        _callback_registry.pop(id(py_callback), None)
    
    # If we stored a Python function, clean it up too
    if hasattr(func, '__call__'):
        _callback_registry.pop(id(func), None)
    
    # Return results as a dictionary
    return {
        'x': np.array([x_best[0], x_best[1], x_best[2]]),
        'fun': fun_best,
        'nfev': nfev,
        'nit': nit,
        'success': success,
        'final_prime_indices': (final_p_idx_x, final_p_idx_y)
    } 