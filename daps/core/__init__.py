# daps/core/__init__.py
# This file makes daps/core a package.

from .daps import daps_minimize
from .function import DAPSFunction, recursive_fractal_cliff_valley_func, rosenbrock_3d_func, sphere_func, ackley_3d_func, rastrigin_func

# Rename for backwards compatibility
recursive_fractal_cliff_valley = recursive_fractal_cliff_valley_func
rosenbrock_3d = rosenbrock_3d_func
ackley_3d = ackley_3d_func
rastrigin = rastrigin_func
