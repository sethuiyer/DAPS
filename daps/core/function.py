"""
Function handling and validation for DAPS.

This module provides utilities for validating and wrapping user-defined functions
to be used with the DAPS optimization algorithm.
"""
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
        # Extract the function from data before validation
        func = data.get('func')
        if func and 'name' not in data:
            data['name'] = getattr(func, '__name__', 'unnamed_function')
        
        # Determine dimensions if not explicitly provided
        if 'dimensions' not in data:
            # Try to infer from bounds
            bounds = data.get('bounds')
            if bounds:
                if len(bounds) == 2:
                    data['dimensions'] = 1
                elif len(bounds) == 4:
                    data['dimensions'] = 2
                elif len(bounds) == 6:
                    data['dimensions'] = 3
                else:
                    raise ValueError("Bounds must contain 2 values (1D), 4 values (2D), or 6 values (3D)")
            else:
                # Default to 3D for backward compatibility
                data['dimensions'] = 3
        
        super().__init__(**data)
        
        # Set up the wrapper function
        self._wrapped_func = self._create_wrapped_func()
    
    @validator('dimensions')
    def validate_dimensions(cls, v):
        """Validate that dimensions are valid."""
        if v not in (1, 2, 3):
            raise ValueError("Dimensions must be 1, 2, or 3")
        return v
    
    @validator('func')
    def validate_function(cls, v, values):
        """Validate that the function is callable with the right signature."""
        if not callable(v):
            raise ValueError("func must be a callable")
        
        # Get dimensions from values (if available)
        dimensions = values.get('dimensions', 3)  # Default to 3D if not yet set
        
        # Try to call with the appropriate number of arguments
        try:
            args = [0.0] * dimensions
            if dimensions == 1:
                result = v(args[0])
            elif dimensions == 2:
                result = v(args[0], args[1])
            else:  # dimensions == 3
                result = v(args[0], args[1], args[2])
                
            if not isinstance(result, (int, float)):
                raise ValueError("Function must return a numeric value")
        except TypeError:
            # Try to call with a single array argument
            try:
                result = v(np.array([0.0] * dimensions))
                if not isinstance(result, (int, float)):
                    raise ValueError("Function must return a numeric value")
            except Exception:
                raise ValueError(
                    f"Function must accept either {dimensions} arguments or "
                    f"a single array argument of length {dimensions}"
                )
        
        return v
    
    @validator('bounds')
    def validate_bounds(cls, v, values):
        """Validate that bounds are properly formatted."""
        if v is None:
            return v
        
        # Get dimensions from values
        dimensions = values.get('dimensions', 3)  # Default to 3D if not yet set
        
        # Check bounds length
        expected_length = dimensions * 2
        if len(v) != expected_length:
            raise ValueError(f"For {dimensions}D optimization, bounds must contain {expected_length} values")
        
        # Check min/max order for each dimension
        for i in range(dimensions):
            if v[i*2] >= v[i*2+1]:
                dim_names = ['x', 'y', 'z']
                dim = dim_names[i]
                raise ValueError(f"{dim}min must be less than {dim}max")
        
        return v
    
    @validator('true_optimum')
    def validate_true_optimum(cls, v, values):
        """Validate that true_optimum has the right size."""
        if v is None:
            return v
        
        dimensions = values.get('dimensions', 3)
        if len(v) != dimensions:
            raise ValueError(f"true_optimum must have length equal to dimensions ({dimensions})")
        
        return v
    
    def _create_wrapped_func(self) -> Callable:
        """Create a wrapper for the function that handles different input formats."""
        func = self.func
        dimensions = self.dimensions
        
        # Try to determine if the function takes separate args or a single array
        try:
            # Test with separate arguments
            if dimensions == 1:
                func(0.0)
            elif dimensions == 2:
                func(0.0, 0.0)
            else:  # dimensions == 3
                func(0.0, 0.0, 0.0)
            
            # Function accepts individual arguments
            if dimensions == 1:
                def wrapper(x, y=None, z=None):
                    return func(x)
            elif dimensions == 2:
                def wrapper(x, y, z=None):
                    return func(x, y)
            else:  # dimensions == 3
                def wrapper(x, y, z):
                    return func(x, y, z)
            
        except TypeError:
            # Try with a single array argument
            try:
                func(np.array([0.0] * dimensions))
                
                # Function accepts array
                def wrapper(x, y=None, z=None):
                    if dimensions == 1:
                        return func(np.array([x]))
                    elif dimensions == 2 and y is not None:
                        return func(np.array([x, y]))
                    elif dimensions == 3 and y is not None and z is not None:
                        return func(np.array([x, y, z]))
                    else:
                        raise ValueError(f"Function requires {dimensions} arguments")
                
            except Exception as e:
                raise ValueError(f"Function has an invalid signature: {e}")
        
        return wrapper
    
    def __call__(self, *args):
        """Call the wrapped function with appropriate number of arguments."""
        if len(args) == 1:
            return self._wrapped_func(args[0])
        elif len(args) == 2:
            return self._wrapped_func(args[0], args[1])
        elif len(args) == 3:
            return self._wrapped_func(args[0], args[1], args[2])
        else:
            raise ValueError(f"Expected 1-3 arguments, got {len(args)}")
    
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