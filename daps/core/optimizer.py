"""
DAPS - Dimensionally Adaptive Prime Search Core Optimizer

A high-performance optimization algorithm for functions of 1-3 dimensions.
"""
import numpy as np
from .function import DAPSFunction
from . import _daps  # This will be the compiled Cython module

def daps_minimize(fun, bounds=None, options=None):
    """
    Minimizes a function using the DAPS algorithm, supporting 1D, 2D, and 3D optimization.

    Args:
        fun: The objective function to minimize. Can be:
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

    # Call the Cython function
    result = _daps.daps_cython(
        daps_func,  # Pass the DAPSFunction instance
        bounds,
        options=options,
        callback=options['callback']
    )

    # Convert to NumPy array for consistency
    result['x'] = np.array(result['x'])
    
    # Add function info if available
    if hasattr(fun, 'info') and callable(fun.info):
        result['function_info'] = fun.info()
    
    return result 