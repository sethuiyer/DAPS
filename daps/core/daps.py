# daps/core/daps.py

"""
DAPS - Dimensionally Adaptive Prime Search

A high-performance global optimization algorithm for functions of 1-3 dimensions.
"""
import numpy as np
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
from .function import DAPSFunction
from . import _daps  # Import the Cython module

def daps_minimize(func: Union[Callable, DAPSFunction], bounds: Optional[List[float]] = None, 
                  options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            If None and func is a DAPSFunction with defined bounds, those bounds will be used.
        options: A dictionary of options:
            'max_iterations': Maximum number of iterations (default: 1000).
            'maxiter': Alias for max_iterations.
            'min_prime_idx_x': Minimum prime index for x dimension (default: 0).
            'min_prime_idx_y': Minimum prime index for y dimension (default: 0).
            'min_prime_idx_z': Minimum prime index for z dimension (default: 0).
            'callback': Callback function (called each iteration).
            'tolerance': Tolerance for termination (default: 1e-8).
            'tol': Alias for tolerance.
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
    # Process options and handle aliases
    if options is None:
        options = {}
    
    # Handle common option aliases
    if 'maxiter' in options and 'max_iterations' not in options:
        options['max_iterations'] = options.pop('maxiter')
    
    if 'tol' in options and 'tolerance' not in options:
        options['tolerance'] = options.pop('tol')
    
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
    dimensions = options.get('dimensions')
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
    
    # Call the Cython wrapper function
    try:
        result = _daps.daps_optimize_wrapper(func, bounds, options)
        
        # Add function info if available
        if hasattr(func, 'info') and callable(func.info):
            result['function_info'] = func.info()
            
        return result
    except Exception as e:
        # Provide more user-friendly error message
        raise RuntimeError(f"DAPS optimization failed: {str(e)}")
