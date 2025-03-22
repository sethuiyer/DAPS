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
        func: The function to be optimized, should take 3 arguments (x, y, z) or a single array argument.
        name: Optional name for the function (defaults to function's __name__).
        bounds: Optional default bounds for the function as (xmin, xmax, ymin, ymax, zmin, zmax).
        true_optimum: Optional known true optimum point (x, y, z) if known, for testing.
        true_value: Optional known function value at the true optimum, if known.
        description: Optional description of the function.
    """
    func: Callable
    name: str = None
    bounds: Optional[Tuple[float, float, float, float, float, float]] = None
    true_optimum: Optional[Tuple[float, float, float]] = None
    true_value: Optional[float] = None
    description: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        # Extract the function from data before validation
        func = data.get('func')
        if func and 'name' not in data:
            data['name'] = getattr(func, '__name__', 'unnamed_function')
        
        super().__init__(**data)
        
        # Set up the wrapper function
        self._wrapped_func = self._create_wrapped_func()
    
    @validator('func')
    def validate_function(cls, v):
        """Validate that the function is callable with the right signature."""
        if not callable(v):
            raise ValueError("func must be a callable")
        
        # Try to call with 3 arguments (x, y, z)
        try:
            result = v(0.0, 0.0, 0.0)
            if not isinstance(result, (int, float)):
                raise ValueError("Function must return a numeric value")
        except TypeError:
            # Try to call with a single array argument
            try:
                result = v(np.array([0.0, 0.0, 0.0]))
                if not isinstance(result, (int, float)):
                    raise ValueError("Function must return a numeric value")
            except Exception:
                raise ValueError(
                    "Function must accept either three arguments (x, y, z) "
                    "or a single array argument of length 3"
                )
        
        return v
    
    @validator('bounds')
    def validate_bounds(cls, v):
        """Validate that bounds are properly formatted."""
        if v is None:
            return v
        
        if len(v) != 6:
            raise ValueError("Bounds must be a tuple of 6 values: (xmin, xmax, ymin, ymax, zmin, zmax)")
        
        for i in range(0, 6, 2):
            if v[i] >= v[i+1]:
                dim = ['x', 'y', 'z'][i//2]
                raise ValueError(f"{dim}min must be less than {dim}max")
        
        return v
    
    def _create_wrapped_func(self) -> Callable:
        """Create a wrapper for the function that handles different input formats."""
        func = self.func
        
        # Try to determine if the function takes 3 args or a single array
        try:
            # Test with 3 separate arguments
            func(0.0, 0.0, 0.0)
            
            # Function accepts (x, y, z)
            def wrapper(x, y, z):
                return func(x, y, z)
            
        except TypeError:
            # Try with a single array argument
            try:
                func(np.array([0.0, 0.0, 0.0]))
                
                # Function accepts array
                def wrapper(x, y, z):
                    return func(np.array([x, y, z]))
                
            except Exception as e:
                raise ValueError(f"Function has an invalid signature: {e}")
        
        return wrapper
    
    def __call__(self, x, y, z):
        """Call the wrapped function."""
        return self._wrapped_func(x, y, z)
    
    def info(self) -> Dict[str, Any]:
        """Return information about the function."""
        return {
            "name": self.name,
            "bounds": self.bounds,
            "true_optimum": self.true_optimum,
            "true_value": self.true_value,
            "description": self.description
        } 