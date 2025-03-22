# daps/core/_daps.pyx

# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.functional cimport function
from libcpp cimport bool # Use standard bool from libcpp

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
        bool success  # Use standard C++ bool
        int final_prime_idx_x
        int final_prime_idx_y
        int final_prime_idx_z
        int dimensions

    # DAPS Optimization function declaration.  VERY IMPORTANT: use correct function pointer type!
    DAPSResult daps_optimize(
        objective_func_ptr func,  # Use the typedef-ed function pointer type
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

    # Built-in test functions (now declared correctly)
    double recursive_fractal_cliff_valley(double x, double y, double z) nogil
    double rosenbrock_3d(double x, double y, double z) nogil
    double sphere_function(double x, double y, double z) nogil
    double ackley_function(double x, double y, double z) nogil
    double rastrigin_function(double x, double y, double z) nogil


# -------------------------------------------------------------
# Cython callback function (with GIL)
# -------------------------------------------------------------
cdef extern from *:  # Use extern "C" to avoid name mangling
    """
    #include <Python.h>
    int call_python_callback(const std::vector<double>& x, double fun_val, int evals, void* py_callback_ptr) {
        if (py_callback_ptr == NULL) {
            return 1; // Continue if no callback
        }

        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        PyObject* pFunc = (PyObject*)py_callback_ptr;
        if (!PyCallable_Check(pFunc)) {
            PyGILState_Release(gstate);
            return 0; // Stop if the callback is not callable.
        }

        PyObject* pArgs = PyTuple_New(3);
        PyObject* pXList = PyList_New(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            PyList_SET_ITEM(pXList, i, PyFloat_FromDouble(x[i]));  // Steals reference
        }
        
        PyTuple_SET_ITEM(pArgs, 0, pXList); // Steals reference
        PyTuple_SET_ITEM(pArgs, 1, PyFloat_FromDouble(fun_val));   // Steals reference
        PyTuple_SET_ITEM(pArgs, 2, PyLong_FromLong(evals));  // Steals reference

        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        Py_DECREF(pArgs); // pArgs, pXList, and item in it no longer needed

        int keep_going = 1;  // Default to continue
        if (pResult != NULL) {
            if (PyObject_IsTrue(pResult)) {
                keep_going = 1; // Continue
            } else {
                keep_going = 0; // Stop
            }
            Py_DECREF(pResult);
        } else {
            PyErr_Print();  // Print any errors from the Python callback
            keep_going = 0; // Stop on error
        }
        PyGILState_Release(gstate);
        return keep_going;
    }
    """
    int call_python_callback(const vector[double]& x, double fun_val, int evals, void* py_callback_ptr)


# -------------------------------------------------------------
# Main optimization interface (Cython)
# -------------------------------------------------------------

def daps_minimize(func, bounds=None, options=None):
    """
    Minimizes a function using the DAPS algorithm, supporting 1D, 2D, and 3D optimization.

    Args:
        func: The objective function to minimize. Can be:
            - A DAPSFunction instance
            - A function with signature:
               * f(x) -> float (1D)
               * f(x, y) -> float (2D)
               * f(x, y, z) -> float (3D)
            - A function with signature f(coords) -> float where coords is a numpy array
        bounds: Bounds for the variables, specified as:
            - [xmin, xmax] for 1D
            - [xmin, xmax, ymin, ymax] for 2D
            - [xmin, xmax, ymin, ymax, zmin, zmax] for 3D
            If None and fun is a DAPSFunction with defined bounds, those bounds will be used.
        options: A dictionary of options:
            'max_iterations': Maximum number of iterations (default: 1000).
            'min_prime_idx_x': Minimum prime index for x dimension (default: 0).
            'min_prime_idx_y': Minimum prime index for y dimension (default: 0).
            'min_prime_idx_z': Minimum prime index for z dimension (default: 0).
            'callback': Callback function (called each iteration).
            'tol': Tolerance for termination (default: 1e-8).
            'verbose': Whether to print progress information (default: False).
            'dimensions': Number of dimensions (1, 2, or 3). If not specified,
                          inferred from bounds length.

    Returns:
        A dictionary containing the optimization results:
            'x': NumPy array of best solution found (length depends on dimensions).
            'fun': Best function value found.
            'nfev': Number of function evaluations.
            'nit': Number of iterations performed.
            'success': Boolean indicating success.
            'dimensions': Number of dimensions used for optimization.
            'final_prime_idx_x': Final prime index for x dimension.
            'final_prime_idx_y': Final prime index for y dimension (if dimensions >= 2).
            'final_prime_idx_z': Final prime index for z dimension (if dimensions >= 3).
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
        'dimensions': None
    }

    # Update defaults with user-provided options
    if options is not None:
        default_options.update(options)
    options = default_options  # Use the merged options

    # Process the function
    if isinstance(func, DAPSFunction):
        # If function is already a DAPSFunction instance, use it directly
        daps_func = func
        # Use the DAPSFunction's bounds if none provided
        if bounds is None and daps_func.bounds is not None:
            bounds = daps_func.bounds
    else:
        # Wrap the function in a DAPSFunction
        try:
            daps_func = DAPSFunction(func=func)
        except Exception as e:
            raise ValueError(f"Invalid function: {e}")

    # Check bounds
    if bounds is None:
        raise ValueError("bounds must be specified either in the function call or in the DAPSFunction")

    # Determine dimensions from bounds if not specified
    dimensions = options['dimensions']
    if dimensions is None:
        # Auto-detect dimensions based on bounds length
        if len(bounds) == 2:
            dimensions = 1
        elif len(bounds) == 4:
            dimensions = 2
        elif len(bounds) == 6:
            dimensions = 3
        else:
            raise ValueError("bounds must contain 2 values (1D), 4 values (2D), or 6 values (3D)")
        options['dimensions'] = dimensions
    else:
        # Validate dimensions
        if dimensions not in (1, 2, 3):
            raise ValueError("dimensions must be 1, 2, or 3")

        # Validate bounds length
        if len(bounds) != dimensions * 2:
            raise ValueError(f"For {dimensions}D optimization, bounds must contain {dimensions*2} values")

    # Input validation (options)
    for key, value in options.items():
        if key not in default_options:
            raise ValueError(f"Invalid option: {key}")
        if key in ('max_iterations', 'min_prime_idx_x', 'min_prime_idx_y', 'min_prime_idx_z'):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        if key == 'callback' and value is not None and not callable(value):
            raise ValueError("callback must be a callable function")
        if key == "tolerance" and value is not None and not isinstance(value, (int, float)):
            raise ValueError("tolerance must be a numeric value")

    # Prepare arguments for the C++ function
    cdef double x_min = bounds[0]
    cdef double x_max = bounds[1]
    cdef double y_min = 0.0
    cdef double y_max = 0.0
    cdef double z_min = 0.0
    cdef double z_max = 0.0

    if dimensions >= 2:
        y_min = bounds[2]
        y_max = bounds[3]
    if dimensions == 3:
        z_min = bounds[4]
        z_max = bounds[5]

    cdef objective_func_ptr cpp_func

    if func is recursive_fractal_cliff_valley_func:
        cpp_func = recursive_fractal_cliff_valley
    elif func is rosenbrock_3d_func:
        cpp_func = rosenbrock_3d
    elif func is sphere_func:
        cpp_func = sphere_function
    elif func is ackley_3d_func:
        cpp_func = ackley_function
    elif func is rastrigin_func:
        cpp_func = rastrigin_function
    # NEW: Use the DAPSFunction wrapper for other functions
    else:
      if dimensions == 1:
          cpp_func = daps_func._wrapped_func
      elif dimensions == 2:
          cpp_func = daps_func._wrapped_func
      else: # dimensions == 3
          cpp_func = daps_func._wrapped_func

    cdef void* callback_ptr = <void*>options['callback'] if options['callback'] is not None else NULL

    # Call the C++ optimization function
    cdef DAPSResult res = daps_optimize(
        cpp_func,
        x_min, x_max,
        y_min, y_max,
        z_min, z_max,
        options['max_iterations'],
        options['min_prime_idx_x'],
        options['min_prime_idx_y'],
        options['min_prime_idx_z'],
        callback_ptr,
        options['tolerance'],
        dimensions
    )

    # Convert C++ vector to NumPy array
    result_x = np.array(res.x)

    # Return results as a dictionary
    result = {
        'x': result_x,
        'fun': res.fun_val,
        'nfev': res.nfev,
        'nit': res.nit,
        'success': res.success,
        'dimensions': dimensions,
        'final_prime_idx_x': res.final_prime_idx_x,
        'final_prime_idx_y': res.final_prime_idx_y,
        'final_prime_idx_z': res.final_prime_idx_z,
    }

    # Add function info if available
    if hasattr(func, 'info') and callable(func.info):
        result['function_info'] = func.info()

    return result
