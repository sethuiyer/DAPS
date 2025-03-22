# daps.pyx
# distutils: language = c++

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector

cdef extern from "daps.cpp":
    cdef vector[double] daps(double (*f)(double, double, double),
                            vector[double]& bounds,
                            int max_iters,
                            int min_prime_idx,
                            int max_prime_idx,
                            void* callback_func,
                            double tol)
    double recursive_fractal_cliff_valley(double x, double y, double z)

# Define a Python wrapper for the C function
def built_in_recursive_fractal_cliff_valley_python(x, y, z):
    return recursive_fractal_cliff_valley(x, y, z)

# Dictionary to store user-defined function instances
# This will prevent the Python objects from being garbage collected
cdef dict _user_functions = {}

# C callback for Python functions
cdef double python_function_callback(double x, double y, double z) with gil:
    try:
        # Get the Python function from the global dictionary
        func = _user_functions['current_func']
        # Call the Python function
        result = func(x, y, z)
        return result
    except Exception as e:
        print(f"Error in Python function callback: {e}")
        return float('inf')  # Return a large value on error

# Function pointer for the C callback
cdef double (*py_func_ptr)(double, double, double) = python_function_callback

def daps_cython(f, bounds, max_iters=60, min_prime_idx=5, max_prime_idx=9, callback=None, tol=None):
    cdef vector[double] bounds_vec = bounds
    cdef vector[double] result
    cdef void* c_callback = <void*>callback if callback is not None else NULL
    cdef double c_tol = tol if tol is not None else -1.0

    # Check if function is the built-in test function
    if f is built_in_recursive_fractal_cliff_valley_python:
        # Use the C++ implementation directly for the built-in function
        result = daps(recursive_fractal_cliff_valley, bounds_vec, max_iters, min_prime_idx, max_prime_idx, c_callback, c_tol)
    else:
        # For user-defined functions, use the Python callback mechanism
        # Store the function in the global dictionary to prevent garbage collection
        _user_functions['current_func'] = f
        
        # Call the C++ implementation with the Python callback
        result = daps(py_func_ptr, bounds_vec, max_iters, min_prime_idx, max_prime_idx, c_callback, c_tol)

    return {
        'x': result[0],
        'y': result[1],
        'z': result[2],
        'fun': result[3],
        'nfev': int(result[4]),
        'nit': int(result[5]),
        'success': True,
        'message': "Optimization terminated successfully.",
        'p_indices': [int(result[6]), int(result[7]), int(result[8])]  # Final prime indices
    } 