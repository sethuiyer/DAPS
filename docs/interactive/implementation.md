# Interactive Demo Implementation

This page explains how the DAPS interactive demo is implemented using Streamlit.

## Overview

The interactive demo provides a visual and hands-on way to understand how the DAPS algorithm works. It visualizes the optimization process, allowing users to experiment with different parameters and see their effects in real-time.

## Tech Stack

The demo is built with:

- **Streamlit**: A Python framework for creating interactive data apps
- **Matplotlib**: For creating 2D and 3D visualizations
- **NumPy**: For numerical operations
- **DAPS**: The core optimization library

## Code Structure

The main structure of the `app.py` file includes:

```
app.py
├── Custom styling functions
├── 2D function creation utilities
├── Plotting utilities
├── DAPS simulation function
└── Main app function
```

## Key Components

### Simulation Function

The core of the demo is the `simulate_daps_2d` function, which provides a simple implementation of the DAPS algorithm for educational purposes:

```python
def simulate_daps_2d(func, domain, num_iterations, primes):
    """
    Simulate the DAPS algorithm for demonstration purposes.
    """
    x_min, x_max, y_min, y_max = domain
    history = []
    
    # Loop through iterations
    best_x, best_y, best_val = None, None, float('inf')
    current_domain = domain.copy()
    
    for i in range(num_iterations):
        # Use the appropriate prime for this iteration
        idx = min(i, len(primes) - 1)
        p = primes[idx]
        
        # Create a grid
        x = np.linspace(current_domain[0], current_domain[1], p)
        y = np.linspace(current_domain[2], current_domain[3], p)
        
        # Evaluate the function at all grid points
        grid_points = []
        grid_values = []
        local_best_x, local_best_y, local_best_val = None, None, float('inf')
        
        for xi in x:
            for yi in y:
                val = func(xi, yi)
                grid_points.append((xi, yi))
                grid_values.append(val)
                
                if val < local_best_val:
                    local_best_x, local_best_y, local_best_val = xi, yi, val
        
        # Update global best if needed
        if local_best_val < best_val:
            best_x, best_y, best_val = local_best_x, local_best_y, local_best_val
        
        # Shrink the domain around the best point
        shrink_factor = 0.7
        domain_width = current_domain[1] - current_domain[0]
        domain_height = current_domain[3] - current_domain[2]
        
        current_domain = [
            local_best_x - shrink_factor * domain_width / 2,
            local_best_x + shrink_factor * domain_width / 2,
            local_best_y - shrink_factor * domain_height / 2,
            local_best_y + shrink_factor * domain_height / 2
        ]
        
        # Store the iteration history
        history.append({
            'iteration': i,
            'prime': p,
            'grid_points': grid_points.copy(),
            'grid_values': grid_values.copy(),
            'best_point': (local_best_x, local_best_y),
            'best_value': local_best_val,
            'domain': current_domain.copy()
        })
    
    return history, (best_x, best_y, best_val)
```

This function tracks the algorithm's progress at each iteration, storing grid points, function values, and domain changes.

### Visualization Components

The demo includes several visualization components:

1. **2D Contour Plots**: Show the function landscape with grid points and best solutions
2. **Domain Shrinking Animation**: Visualizes how the search domain shrinks around promising regions
3. **3D Surface Plot**: Provides a three-dimensional view of the function and optimization path

### Interactive Controls

The sidebar contains controls for:

- **Test Function Selection**: Choose from various benchmark functions
- **Domain Bounds**: Set the search space boundaries
- **Algorithm Parameters**: Configure prime numbers and iteration count

## Extending the Demo

To extend the demo with new features:

1. **Add new test functions**:
   ```python
   def my_new_function(x, y, z):
       return # Your function implementation
   
   # Add to the function selection
   function_type = st.sidebar.selectbox(
       "Test Function", 
       ["My New Function", "Rosenbrock", ...]
   )
   ```

2. **Add new visualization types**:
   ```python
   def plot_my_new_visualization(ax, func, domain, history):
       # Your visualization code here
   
   # Add a new tab
   tab1, tab2, tab3, new_tab = st.tabs(["DAPS Visualization", "3D View", 
                                      "Optimization Results", "My New Visualization"])
   
   with new_tab:
       # Create your visualization
   ```

## Deployment

The demo can be deployed to:

1. **Local environment**: Run with `streamlit run app.py`
2. **Streamlit Cloud**: Deploy directly from a GitHub repository
3. **Custom server**: Deploy using Docker or other container technologies

## Performance Considerations

For optimal performance:

- Use caching for expensive computations with `@st.cache_data`
- Limit the grid resolution for complex functions
- Consider using session state to track user interactions efficiently

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Matplotlib 3D Plotting](https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html)
- [DAPS Documentation](https://github.com/sethuiyer/DAPS) 