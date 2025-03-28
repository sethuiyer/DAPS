# daps/core/function.py

from typing import Callable, Optional, Dict, Any, List, Tuple, Union
import numpy as np
from pydantic import BaseModel, validator, Field

class DAPSFunction(BaseModel):
    """
    A model for validating and wrapping functions for use with DAPS.

    Attributes:
        func: The function to be optimized. Can accept:
            - 1D: f(x)
            - 2D: f(x, y)
            - 3D: f(x, y, z)
            - Or a single array argument of corresponding length
        name: Optional name for the function (defaults to function's __name__).
        bounds: Optional default bounds for the function:
            - 1D: (xmin, xmax)
            - 2D: (xmin, xmax, ymin, ymax)
            - 3D: (xmin, xmax, ymin, ymax, zmin, zmax)
        dimensions: Number of dimensions (1, 2, or 3). If not specified,
                   inferred from bounds length or function signature.
        true_optimum: Optional known true optimum point, with length matching dimensions.
        true_value: Optional known function value at the true optimum, if known.
        description: Optional description of the function.
    """
    func: Callable
    name: str = None
    bounds: Optional[Tuple] = None
    dimensions: Optional[int] = None
    true_optimum: Optional[Tuple] = None
    true_value: Optional[float] = None
    description: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)  # Initialize first
        self._wrapped_func = self._create_wrapped_func()

    @validator('dimensions')
    def validate_dimensions(cls, v):
        if v not in (1, 2, 3):
            raise ValueError("Dimensions must be 1, 2, or 3")
        return v

    @validator('func')
    def validate_function(cls, v, values):
        if not callable(v):
            raise ValueError("func must be a callable")
        dimensions = values.get('dimensions', 3)
        try:
            if dimensions == 1:
                v(0.0)
            elif dimensions == 2:
                v(0.0, 0.0)
            elif dimensions == 3:
                v(0.0, 0.0, 0.0)
        except (TypeError, ValueError):
            try:
                v(np.zeros(dimensions))
            except Exception:
                raise ValueError("Function must accept either separate arguments or a single array")
        return v


    @validator('bounds')
    def validate_bounds(cls, v, values):
        if v is None:
            return None

        dimensions = values.get('dimensions', 3)
        expected_length = 2 * dimensions
        if len(v) != expected_length:
            raise ValueError(f"Bounds must contain {expected_length} values for {dimensions}D.")

        for i in range(dimensions):
            if v[2*i] >= v[2*i + 1]:
                raise ValueError(f"Lower bound must be less than upper bound for dimension {i+1}.")
        return v

    @validator('true_optimum')
    def validate_true_optimum(cls, v, values):
        if v is not None:
            dimensions = values.get('dimensions', 3)
            if len(v) != dimensions:
                raise ValueError("true_optimum must have the same length as dimensions.")
        return v
    

    def _create_wrapped_func(self):
        """Create a wrapper for the function that handles different input formats."""
        func = self.func
        dimensions = self.dimensions
        
        if dimensions == 1:
            def wrapper(x, y=None, z=None):  # y and z are ignored
                return func(x)
        elif dimensions == 2:
            def wrapper(x, y, z=None): # z is ignored
                return func(x, y)
        else:  # dimensions == 3
            def wrapper(x, y, z):
                return func(x, y, z)
        return wrapper

    def __call__(self, *args):
        """Call the wrapped function with appropriate number of arguments."""
        if self.dimensions == 1:
            return self._wrapped_func(args[0])
        elif self.dimensions == 2:
             return self._wrapped_func(args[0], args[1])
        else:
            return self._wrapped_func(args[0], args[1], args[2])

    def info(self) -> Dict[str, Any]:
        """Return information about the function."""
        return {
            "name": self.name,
            "bounds": self.bounds,
            "dimensions": self.dimensions,
            "true_optimum": self.true_optimum,
            "true_value": self.true_value,
            "description": self.description
        }



# Built-in test functions (these are now standard Python functions)
def recursive_fractal_cliff_valley_func(x: float, y: float, z: float) -> float:
    alpha = 2.5
    beta = 0.7
    gamma = 5.0
    delta = 3.0
    
    return (np.sin(alpha * np.power(x + np.pi, 2)) + 
            np.exp(beta * np.abs(y - np.e)) +
            gamma / (1 + np.power(np.power(z - np.sqrt(5.0), 4), 1)) +
            delta * np.sin(10 * np.sin(x * y * z / 100.0)))

recursive_fractal_cliff_valley = DAPSFunction(
    func=recursive_fractal_cliff_valley_func,
    name="Recursive Fractal Cliff Valley",
    bounds=(-15, 5, -5, 15, 0, 10),
    dimensions=3,
    true_optimum=(-np.pi, np.e, np.sqrt(5)),
    true_value=None,
    description="A challenging 3D test function with multiple local minima, cliffs, and valleys."
)
recursive_fractal_cliff_valley.true_value = recursive_fractal_cliff_valley(*recursive_fractal_cliff_valley.true_optimum)


def rosenbrock_3d_func(x: float, y: float, z: float) -> float:
    return 100 * (y - x**2)**2 + (1 - x)**2 + 100 * (z - y**2)**2 + (1 - y)**2

rosenbrock_3d = DAPSFunction(
    func=rosenbrock_3d_func,
    name="Rosenbrock 3D",
    bounds=(-5, 5, -5, 5, -5, 5),
    dimensions=3,
    true_optimum=(1, 1, 1),
    true_value=0.0,
    description="3D extension of the classic Rosenbrock function."
)

def sphere_func(coords):
    return np.sum(coords**2)

sphere = DAPSFunction(
    func=sphere_func,
    name="Sphere Function",
    bounds=(-10, 10, -10, 10, -10, 10),
    dimensions=3,
    true_optimum=(0, 0, 0),
    true_value=0.0,
    description="Simple sphere function (sum of squares)."
)

def ackley_3d_func(x: float, y: float, z: float) -> float:
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq = (x**2 + y**2 + z**2) / 3
    sum_cos = (np.cos(c*x) + np.cos(c*y) + np.cos(c*z)) / 3
    return -a * np.exp(-b * np.sqrt(sum_sq)) - np.exp(sum_cos) + a + np.exp(1)

ackley_3d = DAPSFunction(
    func=ackley_3d_func,
    name="Ackley 3D",
    bounds=(-5, 5, -5, 5, -5, 5),
    dimensions=3,
    true_optimum=(0, 0, 0),
    true_value=0.0,
    description="3D version of the Ackley function, a highly non-convex test function."
)

def rastrigin_func(coords):
    A = 10
    n = len(coords)
    return A * n + np.sum(coords**2 - A * np.cos(2 * np.pi * coords))

rastrigin = DAPSFunction(
    func=rastrigin_func,
    name="Rastrigin 3D",
    bounds=(-5.12, 5.12, -5.12, 5.12, -5.12, 5.12),
    dimensions=3,
    true_optimum=(0, 0, 0),
    true_value=0.0,
    description="3D Rastrigin function, a highly multimodal test function."
)
