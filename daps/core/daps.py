# daps/core/daps.py

from ._daps import daps_cython
from .function import DAPSFunction

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

    # Default options, now handled primarily in Cython
    default_options = {
        'dimensions': None,  # Let Cython determine it from the bounds
        'min_prime_idx_x': 0,
        'min_prime_idx_y': 0,
        'min_prime_idx_z': 0,
        'max_iterations': 1000,
        'tolerance': 1e-8,
        'callback': None,
        'verbose': False
    }

    # Update defaults with user-provided options
    if options is not None:
        default_options.update(options)
    options = default_options  # Use the merged options

    # Call the Cython function
    result = daps_cython(
        func=daps_func,
        bounds=bounds,
        options=options
    )
    
    # Add function info if available
    if hasattr(fun, 'info') and callable(fun.info):
        result['function_info'] = fun.info()

    return result
