"""
DAPS - Dimensionally Adaptive Prime Search Core Optimizer

A high-performance optimization algorithm for 3D functions.
"""
import numpy as np
from .function import DAPSFunction
from . import _daps  # This will be the compiled Cython module

def daps_minimize(fun, bounds=None, options=None):
    """
    Minimizes a 3D function using the DAPS algorithm.

    Args:
        fun: The objective function to minimize. Can be:
            - A DAPSFunction instance
            - A function with signature f(x, y, z) -> float
            - A function with signature f(coords) -> float where coords is a numpy array
        bounds: A list/tuple of (xmin, xmax, ymin, ymax, zmin, zmax).
            If None and fun is a DAPSFunction with defined bounds, those bounds will be used.
        options: A dictionary of options:
            'maxiter': Maximum number of iterations (default: 60).
            'min_prime_idx': Minimum prime index (default: 5).
            'max_prime_idx': Maximum prime index (default: 9).
            'callback': Callback function (called each iteration).
            'tol': Tolerance for termination (change in best function value).

    Returns:
        A dictionary containing the optimization results:
            'x': NumPy array [x_best, y_best, z_best] - Best solution found.
            'fun': Best function value found.
            'nfev': Number of function evaluations.
            'nit': Number of iterations performed.
            'success': Boolean indicating success.
            'message': Success or error message.
            'p_indices': Final prime indices used.
    """
    # Default options
    default_options = {
        'maxiter': 60,
        'min_prime_idx': 5,
        'max_prime_idx': 9,
        'callback': None,
        'tol': None,
    }
    
    # Update defaults with user-provided options
    if options is not None:
        default_options.update(options)
    options = default_options  # Use the merged options

    # Process the function
    if isinstance(fun, DAPSFunction):
        # If function is already a DAPSFunction instance, use it directly
        daps_func = fun
        # Use the DAPSFunction's bounds if none provided
        if bounds is None and daps_func.bounds is not None:
            bounds = daps_func.bounds
    else:
        # Wrap the function in a DAPSFunction
        try:
            daps_func = DAPSFunction(func=fun)
        except Exception as e:
            raise ValueError(f"Invalid function: {e}")
    
    # Check bounds
    if bounds is None:
        raise ValueError("bounds must be specified either in the function call or in the DAPSFunction")
    
    # Input validation (bounds)
    if len(bounds) != 6:
        raise ValueError("bounds must be a list/tuple of length 6: (xmin, xmax, ymin, ymax, zmin, zmax)")
    if not all(isinstance(b, (int, float)) for b in bounds):
        raise ValueError("bounds must contain numeric values")

    # Input validation (options)
    for key, value in options.items():
        if key not in default_options:
            raise ValueError(f"Invalid option: {key}")
        if key in ('maxiter', 'min_prime_idx', 'max_prime_idx'):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        if key == 'callback' and value is not None and not callable(value):
            raise ValueError("callback must be a callable function.")
        if key == "tol" and value is not None and not isinstance(value, (int, float)):
            raise ValueError("tol must be a numeric value")

    # Call the Cython function
    result = _daps.daps_cython(
        daps_func,  # Pass the DAPSFunction instance
        bounds,
        max_iters=options['maxiter'],
        min_prime_idx=options['min_prime_idx'],
        max_prime_idx=options['max_prime_idx'],
        callback=options['callback'],
        tol=options['tol']
    )

    # Convert to NumPy array for consistency
    result['x'] = np.array([result.pop('x'), result.pop('y'), result.pop('z')])
    
    # Add function info if available
    if isinstance(fun, DAPSFunction):
        result['function_info'] = fun.info()
    
    return result 