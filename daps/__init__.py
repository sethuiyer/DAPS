"""
DAPS - Dimensionally Adaptive Prime Search

A high-performance optimization algorithm for 3D functions, implemented in C++/Cython.
"""

__version__ = '0.1.0'

# Import the main functions
from daps.core.optimizer import daps_minimize
from daps.core.function import DAPSFunction

# Import built-in test functions
from daps.core.test_functions import (
    recursive_fractal_cliff_valley,
    rosenbrock_3d,
    sphere,
    ackley_3d,
    rastrigin
)

__all__ = [
    # Main API
    'daps_minimize',
    'DAPSFunction',
    
    # Built-in test functions
    'recursive_fractal_cliff_valley',
    'rosenbrock_3d',
    'sphere',
    'ackley_3d',
    'rastrigin'
]
