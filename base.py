import numpy as np
from sympy import primerange

##############################################################################
# 1. UTILITIES & CLASSES
##############################################################################

class DAPSFunction:
    """
    A container for user-defined functions or built-in test functions
    for the DAPS optimizer.

    Attributes:
    -----------
    func : callable
        The function to optimize. For an N-dimensional problem, it must
        accept N parameters, e.g. func(x, y, z).

    name : str
        A human-readable name for the function.

    bounds : list or array
        A flat list that defines the domain: [xmin, xmax, ymin, ymax, ...].
        Must match the dimension.

    dimensions : int
        Number of dimensions in this optimization problem.

    true_optimum : list or array, optional
        Known optimal point(s) (for reference).

    true_value : float, optional
        Known minimum function value (for reference).

    description : str, optional
        Additional details about the function.
    """
    def __init__(self, func, name, bounds, dimensions,
                 true_optimum=None, true_value=None, description=""):
        self.func = func
        self.name = name
        self.bounds = bounds
        self.dimensions = dimensions
        self.true_optimum = true_optimum
        self.true_value = true_value
        self.description = description


##############################################################################
# 2. CORE DAPS ALGORITHM
##############################################################################

def daps_minimize(daps_func, options=None):
    """
    Perform the Dimensionally Adaptive Prime Search (DAPS) on a given function.

    Parameters
    ----------
    daps_func : DAPSFunction
        The wrapped function to optimize. Must contain .func (callable),
        .bounds (list), .dimensions (int).

    options : dict, optional
        Algorithmic parameters:
        - maxiter        : int, maximum iterations (default=100)
        - min_prime_idx  : int, initial prime index (default=0)
        - max_prime_idx  : int, maximum prime index allowed (default=10)
        - tol            : float, tolerance for early stopping (default=1e-6)
        - shrink_factor  : float, proportion of how fast the domain shrinks
                           each iteration (default=0.5)
        - improvement_factor : float, factor threshold to decide prime
                               index up/down (default=0.9)

    Returns
    -------
    result : dict
        {
          'x'   : array, best found solution,
          'fun' : float, best function value,
          'nit' : int, number of iterations actually performed
        }
    """
    if not isinstance(daps_func, DAPSFunction):
        raise ValueError("daps_minimize expects a DAPSFunction object.")

    # Default options
    if options is None:
        options = {}
    maxiter = options.get('maxiter', 100)
    min_prime_idx = options.get('min_prime_idx', 0)
    max_prime_idx = options.get('max_prime_idx', 10)
    tol = options.get('tol', 1e-6)
    alpha = options.get('shrink_factor', 0.5)
    improvement_factor = options.get('improvement_factor', 0.9)

    # Prepare prime list
    # We'll generate enough primes to avoid index overflow
    prime_list = list(primerange(2, 2000))  # can store >10 primes easily
    prime_index = min_prime_idx

    # Extract dimension & define initial domain
    dims = daps_func.dimensions
    # Expecting a flat [x_min, x_max, y_min, y_max, ...]
    domain_array = np.array(daps_func.bounds).reshape(dims, 2)

    # Track global best
    best_x = None
    best_val = np.inf

    # --- ITERATION LOOP ---
    for iteration in range(1, maxiter+1):
        p = prime_list[prime_index]  # current prime

        # Build sampling grid for each dimension
        grid_axes = [np.linspace(domain_array[d,0],
                                 domain_array[d,1],
                                 p) for d in range(dims)]
        # Create the full Cartesian product
        mesh = np.meshgrid(*grid_axes)
        samples = np.vstack([m.ravel() for m in mesh]).T  # shape: (#points, dims)

        # Evaluate function at each sample
        # The function must accept *pt if dims>1, or x if dims=1
        if dims == 1:
            evals = np.array([daps_func.func(xi[0]) for xi in samples])
        else:
            evals = np.array([daps_func.func(*pt) for pt in samples])

        # Identify best in this iteration
        idx_best = np.argmin(evals)
        local_best_x = samples[idx_best]
        local_best_val = evals[idx_best]

        # Update global best if improved
        if local_best_val < best_val:
            # Check improvement factor
            ratio_improvement = 0 if best_val == np.inf else (best_val - local_best_val)/abs(best_val)
            best_val = local_best_val
            best_x = local_best_x

        # Early stopping if tolerance reached
        if best_val < tol:
            return {
                'x'   : best_x,
                'fun' : best_val,
                'nit' : iteration
            }

        # ADAPTIVE PRIME INDEX:
        # If local improvement is good -> zoom in (increase prime index)
        # else zoom out (lower prime index)
        # We'll compare local_best_val with the improvement
        # or just use ratio_improvement if we changed best
        if best_val < np.inf and ratio_improvement > (1 - improvement_factor):
            prime_index = min(prime_index + 1, max_prime_idx)
        else:
            prime_index = max(prime_index - 1, min_prime_idx)

        # ADAPTIVE SHRINK of domain around best_x
        for d in range(dims):
            current_span = domain_array[d, 1] - domain_array[d, 0]
            domain_array[d, 0] = best_x[d] - alpha * current_span / 2
            domain_array[d, 1] = best_x[d] + alpha * current_span / 2

    # If we exhausted all iterations
    return {
        'x'   : best_x,
        'fun' : best_val,
        'nit' : maxiter
    }

##############################################################################
# 3. EXAMPLE USAGE
##############################################################################

if __name__ == "__main__":
    # You can do a quick self-test or usage sample here:
    # 1D test function
    def parabola_1d(x):
        return (x - 2)**2

    # 2D test
    def himmelblau_2d(x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    # 3D custom
    import math
    def custom_3d_func(x, y, z):
        return math.sin(x*y) + math.cos(y*z) + x*x + y*y + z*z

    # Create DAPSFunction objects
    parabola = DAPSFunction(parabola_1d,
        "Parabola 1D",
        bounds=[-10,10],
        dimensions=1
    )
    himmelblau = DAPSFunction(himmelblau_2d,
        "Himmelblau 2D",
        bounds=[-5,5, -5,5],
        dimensions=2
    )
    custom_3d = DAPSFunction(custom_3d_func,
        "Custom 3D",
        bounds=[-5,5, -5,5, -5,5],
        dimensions=3
    )

    print("\n--- Minimizing Parabola 1D ---")
    r1 = daps_minimize(parabola, {'maxiter':50})
    print(r1)

    print("\n--- Minimizing Himmelblau 2D ---")
    r2 = daps_minimize(himmelblau, {'maxiter':80})
    print(r2)

    print("\n--- Minimizing Custom 3D Function ---")
    r3 = daps_minimize(custom_3d, {'maxiter':100})
    print(r3)

