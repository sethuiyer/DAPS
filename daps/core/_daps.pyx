# daps/core/_daps.pyx

# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.functional cimport function
from libcpp cimport bool

from .function import DAPSFunction, recursive_fractal_cliff_valley_func, rosenbrock_3d_func, sphere_func, ackley_3d_func, rastrigin_func


# Define the signatures that the built in test functions must have, this allows the functions to be callable by the optimizer
ctypedef double (*objective_func_ptr)(double, double, double)

# -------------------------------------------------------------
# Extern declarations for C++ functions/structs
# -------------------------------------------------------------
cdef extern from "daps.cpp":
    # DAPSResult struct (make it match the C++ definition)
    cdef struct DAPSResult:
        vector[double] x
        double fun_val
        int nfev
        int nit
        bool success
        int final_prime_idx_x
        int final_prime_idx_y
        int final_prime_idx_z
        int dimensions

    # DAPS Optimization function declaration
    DAPSResult daps_optimize(
        objective_func_ptr func,
        double x_min, double x_max,
        double y_min, double y_max,
        double z_min, double z_max,
        int max_iter,
        int min_prime_idx_x,
        int min_prime_idx_y,
        int min_prime_idx_z,
        void* callback_ptr,
        double tol,
        int dimensions
    ) nogil

    # Built-in test functions
    double recursive_fractal_cliff_valley(double x, double y, double z) nogil
    double rosenbrock_3d(double x, double y, double z) nogil
    double sphere_function(double x, double y, double z) nogil
    double ackley_function(double x, double y, double z) nogil
    double rastrigin_function(double x, double y, double z) nogil

# Store user defined functions to prevent garbage collection
cdef dict _user_functions = {}

# C function that will be called by Python - allows Python callback to be called from C++
# Export this function using extern "C" to prevent name mangling
cdef extern "C" bool call_python_callback(const vector[double]& x, double fun_val, int evals, void* py_callback_ptr) with gil:
    """
    Callback bridge function to allow C++ to call into Python
    """
    if py_callback_ptr != NULL:
        # Cast void pointer back to Python object pointer
        callback = <object>py_callback_ptr
        
        # Convert vector to numpy array
        x_array = np.asarray([x[i] for i in range(x.size())])
        
        # Call the Python callback
        try:
            result = callback(x_array, fun_val, evals)
            if result is not None:
                return result
            return True
        except Exception as e:
            print(f"Error in Python callback: {e}")
            return False
    return True

# Wrapper functions for built-in test functions
def py_recursive_fractal_cliff_valley(double x, double y, double z):
    return recursive_fractal_cliff_valley(x, y, z)

def py_rosenbrock_3d(double x, double y, double z):
    return rosenbrock_3d(x, y, z)

def py_sphere_function(double x, double y, double z):
    return sphere_function(x, y, z)

def py_ackley_function(double x, double y, double z):
    return ackley_function(x, y, z)

def py_rastrigin_function(double x, double y, double z):
    return rastrigin_function(x, y, z)

# Main optimization function wrapper
def daps_optimize_wrapper(func, bounds, options=None):
    """
    Python wrapper for the C++ DAPS optimizer
    
    Args:
        func: Function to optimize
        bounds: [xmin, xmax, ymin, ymax, zmin, zmax] bounds
        options: Dictionary of optimizer options
        
    Returns:
        Dictionary with optimization results
    """
    # Default options
    default_options = {
        'max_iterations': 1000,
        'min_prime_idx_x': 0,
        'min_prime_idx_y': 0,
        'min_prime_idx_z': 0,
        'callback': None,
        'tolerance': 1e-8,
        'verbose': False,
        'dimensions': 3
    }
    
    # Update with user options
    if options is not None:
        default_options.update(options)
    
    # Extract options
    cdef int max_iter = default_options['max_iterations']
    cdef int min_prime_idx_x = default_options['min_prime_idx_x']
    cdef int min_prime_idx_y = default_options['min_prime_idx_y']
    cdef int min_prime_idx_z = default_options['min_prime_idx_z']
    cdef double tol = default_options['tolerance']
    cdef int dimensions = default_options['dimensions']
    cdef void* callback_ptr = NULL
    
    # Get callback if provided
    callback = default_options['callback']
    if callback is not None:
        callback_ptr = <void*>callback
    
    # Validate dimensions
    if dimensions < 1 or dimensions > 3:
        raise ValueError("Dimensions must be 1, 2, or 3")
    
    # Validate bounds based on dimensions
    if dimensions == 1 and len(bounds) < 2:
        raise ValueError("For 1D optimization, bounds must be [xmin, xmax]")
    elif dimensions == 2 and len(bounds) < 4:
        raise ValueError("For 2D optimization, bounds must be [xmin, xmax, ymin, ymax]")
    elif dimensions == 3 and len(bounds) < 6:
        raise ValueError("For 3D optimization, bounds must be [xmin, xmax, ymin, ymax, zmin, zmax]")
    
    # Extract bounds
    cdef double x_min = bounds[0]
    cdef double x_max = bounds[1]
    cdef double y_min = 0.0
    cdef double y_max = 0.0
    cdef double z_min = 0.0
    cdef double z_max = 0.0
    
    if dimensions >= 2:
        y_min = bounds[2]
        y_max = bounds[3]
        
    if dimensions >= 3:
        z_min = bounds[4]
        z_max = bounds[5]
    
    # Determine the actual objective function to use
    cdef DAPSResult result
    
    # Special handling for built-in functions
    if func is recursive_fractal_cliff_valley_func or func is py_recursive_fractal_cliff_valley:
        result = daps_optimize(&recursive_fractal_cliff_valley, 
                              x_min, x_max, y_min, y_max, z_min, z_max,
                              max_iter, min_prime_idx_x, min_prime_idx_y, min_prime_idx_z,
                              callback_ptr, tol, dimensions)
    elif func is rosenbrock_3d_func or func is py_rosenbrock_3d:
        result = daps_optimize(&rosenbrock_3d, 
                              x_min, x_max, y_min, y_max, z_min, z_max,
                              max_iter, min_prime_idx_x, min_prime_idx_y, min_prime_idx_z,
                              callback_ptr, tol, dimensions)
    elif func is sphere_func or func is py_sphere_function:
        result = daps_optimize(&sphere_function, 
                              x_min, x_max, y_min, y_max, z_min, z_max,
                              max_iter, min_prime_idx_x, min_prime_idx_y, min_prime_idx_z,
                              callback_ptr, tol, dimensions)
    elif func is ackley_3d_func or func is py_ackley_function:
        result = daps_optimize(&ackley_function, 
                              x_min, x_max, y_min, y_max, z_min, z_max,
                              max_iter, min_prime_idx_x, min_prime_idx_y, min_prime_idx_z,
                              callback_ptr, tol, dimensions)
    elif func is rastrigin_func or func is py_rastrigin_function:
        result = daps_optimize(&rastrigin_function, 
                              x_min, x_max, y_min, y_max, z_min, z_max,
                              max_iter, min_prime_idx_x, min_prime_idx_y, min_prime_idx_z,
                              callback_ptr, tol, dimensions)
    else:
        # For user-defined functions, store the function and use a proxy
        _user_functions['current_func'] = func
        
        # TODO: Implement user-defined function handling
        raise NotImplementedError("User-defined functions are not yet implemented")
    
    # Create return dictionary
    x_result = [result.x[i] for i in range(dimensions)]
    
    return_dict = {
        'x': np.array(x_result),
        'fun': result.fun_val,
        'nfev': result.nfev,
        'nit': result.nit,
        'success': result.success,
        'dimensions': result.dimensions,
        'final_prime_idx_x': result.final_prime_idx_x,
    }
    
    if dimensions >= 2:
        return_dict['final_prime_idx_y'] = result.final_prime_idx_y
        
    if dimensions >= 3:
        return_dict['final_prime_idx_z'] = result.final_prime_idx_z
    
    return return_dict
