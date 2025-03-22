#!/usr/bin/env python3
"""
Interactive DAPS Demo
A Streamlit app that demonstrates the DAPS algorithm with interactive controls.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import time
import sys
import os

# Add the parent directory to the path so we can import DAPS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import DAPS (will show error message if not installed)
try:
    from daps import DAPSFunction, daps_minimize
    from daps.functions import (
        rosenbrock_3d_function,
        ackley_3d_function,
        sphere_3d_function,
        rastrigin_3d_function,
        recursive_fractal_cliff_valley_function
    )
    DAPS_AVAILABLE = True
except ImportError:
    DAPS_AVAILABLE = False

# Set up the page
st.set_page_config(
    page_title="DAPS Interactive Demo",
    page_icon="🎯",
    layout="wide",
)

# Custom colors
DAPS_COLOR = '#4B0082'  # Indigo
HIGHLIGHT_COLOR = '#FF5733'  # Coral

def apply_custom_styles():
    """Apply custom CSS styles to the app."""
    st.markdown("""
    <style>
        .main-title {
            color: #4B0082;
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subtitle {
            font-size: 24px;
            margin-bottom: 1rem;
            text-align: center;
            color: #555;
        }
        .info-box {
            padding: 20px;
            border-radius: 5px;
            background-color: #f8f9fa;
            margin-bottom: 1rem;
        }
        .highlight {
            color: #FF5733;
            font-weight: bold;
        }
        .success {
            color: #28a745;
            font-weight: bold;
        }
        .parameter-title {
            font-weight: bold;
            font-size: 18px;
            margin-top: 1rem;
        }
        .result-title {
            font-weight: bold;
            font-size: 20px;
            color: #4B0082;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        footer {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to create a 2D target function
def create_2d_function(function_type, z_value=0):
    """Create a 2D function by fixing one dimension of a 3D function."""
    if function_type == "Rosenbrock":
        return lambda x, y: rosenbrock_3d_function.func(x, y, z_value)
    elif function_type == "Ackley":
        return lambda x, y: ackley_3d_function.func(x, y, z_value)
    elif function_type == "Sphere":
        return lambda x, y: sphere_3d_function.func(x, y, z_value)
    elif function_type == "Rastrigin":
        return lambda x, y: rastrigin_3d_function.func(x, y, z_value)
    elif function_type == "Recursive Fractal Cliff Valley":
        return lambda x, y: recursive_fractal_cliff_valley_function.func(x, y, z_value)
    else:
        return lambda x, y: x**2 + y**2  # default to sphere function

def plot_2d_function(ax, func, domain):
    """Plot a 2D function as a contour plot with colorbar."""
    x_min, x_max, y_min, y_max = domain
    
    # Create a grid of points
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate the function on the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(X[i, j], Y[i, j])
    
    # Create contour plot
    contour = ax.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Add colorbar
    plt.colorbar(contour, ax=ax, label='Function Value')
    
    return X, Y, Z

def plot_3d_function(ax, func, domain):
    """Plot a 3D surface of the function."""
    x_min, x_max, y_min, y_max = domain
    
    # Create a grid of points
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate the function on the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(X[i, j], Y[i, j])
    
    # Create 3D surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True, edgecolor='none')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    
    return X, Y, Z

def simulate_daps_2d(func, domain, num_iterations, primes):
    """
    Simulate the DAPS algorithm for demonstration purposes.
    In real use, you would call daps_minimize directly.
    This function visualizes the intermediate steps.
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
        
        # Store the grid for visualization
        grid_points = []
        grid_values = []
        
        # Evaluate the function at all grid points
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

def main():
    """Main function for the Streamlit app."""
    apply_custom_styles()
    
    # App header
    st.markdown('<div class="main-title">DAPS Interactive Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Dimensionally Adaptive Prime Search Algorithm</div>', unsafe_allow_html=True)
    
    # Check if DAPS is available
    if not DAPS_AVAILABLE:
        st.error("DAPS package is not installed. Please install it to use this demo.")
        st.markdown("""
        ```bash
        pip install daps
        ```
        """)
        return
    
    # Sidebar for controls
    st.sidebar.title("DAPS Parameters")
    
    # Select test function
    function_type = st.sidebar.selectbox(
        "Test Function", 
        ["Rosenbrock", "Ackley", "Sphere", "Rastrigin", "Recursive Fractal Cliff Valley"]
    )
    
    # Function description
    function_descriptions = {
        "Rosenbrock": "A classic non-convex function with a narrow valley that is difficult to navigate.",
        "Ackley": "A highly non-convex function with many local minima.",
        "Sphere": "A simple convex function for basic testing.",
        "Rastrigin": "A highly multimodal function with many regular local minima.",
        "Recursive Fractal Cliff Valley": "A challenging function with fractal-like structure and discontinuities."
    }
    
    st.sidebar.markdown(f"**Description**: {function_descriptions[function_type]}")
    
    # Fixed z-value for 2D visualization
    z_value = st.sidebar.slider("Z Value (for 2D visualization)", -5.0, 5.0, 0.0)
    
    # Domain bounds
    st.sidebar.markdown("### Domain Bounds")
    x_min = st.sidebar.slider("X Min", -10.0, 0.0, -5.0)
    x_max = st.sidebar.slider("X Max", 0.0, 10.0, 5.0)
    y_min = st.sidebar.slider("Y Min", -10.0, 0.0, -5.0)
    y_max = st.sidebar.slider("Y Max", 0.0, 10.0, 5.0)
    
    domain = [x_min, x_max, y_min, y_max]
    
    # Algorithm parameters
    st.sidebar.markdown("### Algorithm Parameters")
    
    # Define primes
    available_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    min_prime_idx = st.sidebar.slider("Min Prime Index", 0, 10, 2)
    max_prime_idx = st.sidebar.slider("Max Prime Index", min_prime_idx, 19, 6)
    
    # Get the actual primes to use
    used_primes = available_primes[min_prime_idx:max_prime_idx+1]
    st.sidebar.markdown(f"Primes used: {used_primes}")
    
    num_iterations = st.sidebar.slider("Number of Iterations", 1, 10, 3)
    
    # Create the 2D function
    func_2d = create_2d_function(function_type, z_value)
    
    # Main content area
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"""
    This interactive demo shows how the DAPS algorithm works on the **{function_type}** function.
    
    The algorithm uses a sequence of prime numbers ({', '.join(map(str, used_primes))}) to create
    sampling grids, evaluates the function at grid points, and progressively shrinks the domain
    around promising regions.
    
    Adjust the parameters in the sidebar to see how they affect the optimization process.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Set up tabs
    tab1, tab2, tab3 = st.tabs(["DAPS Visualization", "3D View", "Optimization Results"])
    
    with tab1:
        # Run the simulation when requested
        if st.button("Run DAPS Simulation"):
            with st.spinner("Running DAPS simulation..."):
                # Run the DAPS simulation
                history, (best_x, best_y, best_val) = simulate_daps_2d(
                    func_2d, domain, num_iterations, used_primes
                )
                
                # Store in session state to keep it between reruns
                st.session_state.history = history
                st.session_state.best_result = (best_x, best_y, best_val)
        
        # If we have history, visualize the iterations
        if 'history' in st.session_state:
            history = st.session_state.history
            
            # Create plot
            iterations_to_show = min(num_iterations, len(history))
            
            # Create a multi-iteration visualization
            cols = st.columns(min(3, iterations_to_show))
            
            for i, col in enumerate(cols):
                if i < iterations_to_show:
                    iter_data = history[i]
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    plot_2d_function(ax, func_2d, iter_data['domain'])
                    
                    # Plot grid points
                    grid_x = [p[0] for p in iter_data['grid_points']]
                    grid_y = [p[1] for p in iter_data['grid_points']]
                    ax.scatter(grid_x, grid_y, color=DAPS_COLOR, s=50, alpha=0.7)
                    
                    # Highlight best point
                    best_x, best_y = iter_data['best_point']
                    ax.scatter([best_x], [best_y], color=HIGHLIGHT_COLOR, s=150, marker='*', edgecolor='black')
                    
                    # Draw the domain for the next iteration
                    if i < iterations_to_show - 1:
                        next_domain = history[i+1]['domain']
                        rect = plt.Rectangle(
                            (next_domain[0], next_domain[2]),
                            next_domain[1] - next_domain[0],
                            next_domain[3] - next_domain[2],
                            fill=False, edgecolor='red', linestyle='--', linewidth=2
                        )
                        ax.add_patch(rect)
                    
                    ax.set_title(f"Iteration {i+1}, p={iter_data['prime']}")
                    col.pyplot(fig)
                    
                    # Show details for this iteration
                    col.markdown(f"""
                    **Domain**: [{iter_data['domain'][0]:.2f}, {iter_data['domain'][1]:.2f}] × [{iter_data['domain'][2]:.2f}, {iter_data['domain'][3]:.2f}]  
                    **Best Point**: ({best_x:.4f}, {best_y:.4f})  
                    **Best Value**: {iter_data['best_value']:.6f}  
                    **Points Evaluated**: {len(iter_data['grid_points'])}
                    """)
            
            # Show the domain shrinking animation
            st.markdown("### Domain Shrinking Animation")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            X, Y, Z = plot_2d_function(ax, func_2d, domain)
            
            # Add initial domain
            rect = plt.Rectangle(
                (domain[0], domain[2]),
                domain[1] - domain[0],
                domain[3] - domain[2],
                fill=False, edgecolor='white', linestyle='-', linewidth=2,
                label="Initial Domain"
            )
            ax.add_patch(rect)
            
            # Add all iteration domains with different colors
            colors = plt.cm.plasma(np.linspace(0, 1, len(history)))
            
            for i, iter_data in enumerate(history):
                rect = plt.Rectangle(
                    (iter_data['domain'][0], iter_data['domain'][2]),
                    iter_data['domain'][1] - iter_data['domain'][0],
                    iter_data['domain'][3] - iter_data['domain'][2],
                    fill=False, edgecolor=colors[i], linestyle='-', linewidth=2,
                    label=f"Iteration {i+1}"
                )
                ax.add_patch(rect)
                
                # Add best point for this iteration
                best_x, best_y = iter_data['best_point']
                ax.scatter([best_x], [best_y], color=colors[i], s=100, marker='o', edgecolor='black')
            
            ax.legend(loc='upper right')
            ax.set_title("DAPS Domain Shrinking Process")
            
            st.pyplot(fig)
    
    with tab2:
        # Create 3D view of the function
        st.markdown("### 3D Function Visualization")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_function(ax, func_2d, domain)
        
        # If we have history, add the optimization path
        if 'history' in st.session_state:
            history = st.session_state.history
            
            # Plot the optimization path
            path_x = [history[0]['best_point'][0]]
            path_y = [history[0]['best_point'][1]]
            path_z = [history[0]['best_value']]
            
            for iter_data in history[1:]:
                path_x.append(iter_data['best_point'][0])
                path_y.append(iter_data['best_point'][1])
                path_z.append(iter_data['best_value'])
            
            ax.plot(path_x, path_y, path_z, color=HIGHLIGHT_COLOR, linewidth=3, marker='o', markersize=8)
            
            # Highlight final point
            best_x, best_y, best_val = st.session_state.best_result
            ax.scatter([best_x], [best_y], [best_val], color='red', s=200, marker='*', edgecolor='black')
        
        ax.view_init(30, 45)  # Set the viewing angle
        st.pyplot(fig)
        
        # Add view angle controls
        col1, col2 = st.columns(2)
        elevation = col1.slider("Elevation Angle", 0, 90, 30)
        azimuth = col2.slider("Azimuth Angle", 0, 360, 45)
        
        # Update the view angle when requested
        if st.button("Update View Angle"):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            plot_3d_function(ax, func_2d, domain)
            
            # If we have history, add the optimization path
            if 'history' in st.session_state:
                history = st.session_state.history
                
                # Plot the optimization path
                path_x = [history[0]['best_point'][0]]
                path_y = [history[0]['best_point'][1]]
                path_z = [history[0]['best_value']]
                
                for iter_data in history[1:]:
                    path_x.append(iter_data['best_point'][0])
                    path_y.append(iter_data['best_point'][1])
                    path_z.append(iter_data['best_value'])
                
                ax.plot(path_x, path_y, path_z, color=HIGHLIGHT_COLOR, linewidth=3, marker='o', markersize=8)
                
                # Highlight final point
                best_x, best_y, best_val = st.session_state.best_result
                ax.scatter([best_x], [best_y], [best_val], color='red', s=200, marker='*', edgecolor='black')
            
            ax.view_init(elevation, azimuth)
            st.pyplot(fig)
    
    with tab3:
        st.markdown("### Optimization Results")
        
        # Show results if we have them
        if 'best_result' in st.session_state:
            best_x, best_y, best_val = st.session_state.best_result
            
            st.markdown('<div class="result-title">Final Solution:</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("X Coordinate", f"{best_x:.6f}")
            col2.metric("Y Coordinate", f"{best_y:.6f}")
            col3.metric("Function Value", f"{best_val:.6f}")
            
            # Compare with the actual DAPS optimizer
            st.markdown('<div class="result-title">Compare with full DAPS optimizer:</div>', unsafe_allow_html=True)
            
            if st.button("Run Full DAPS Optimization"):
                with st.spinner("Running full DAPS optimization..."):
                    # Create a custom function with fixed z
                    def func_for_daps(x, y, z=None):
                        return func_2d(x, y)
                    
                    custom_func = DAPSFunction(
                        func=func_for_daps,
                        name="Custom 2D Function",
                        bounds=[domain[0], domain[1], domain[2], domain[3], -1, 1]  # Add dummy z bounds
                    )
                    
                    # Run DAPS with the specified primes
                    start_time = time.time()
                    result = daps_minimize(
                        custom_func,
                        options={
                            'maxiter': num_iterations * 2,  # Give it more iterations for better results
                            'min_prime_idx': min_prime_idx,
                            'max_prime_idx': max_prime_idx,
                            'verbose': False
                        }
                    )
                    end_time = time.time()
                    
                    # Store the result
                    st.session_state.full_daps_result = {
                        'x': result['x'][0],
                        'y': result['x'][1],
                        'fun': result['fun'],
                        'nfev': result['nfev'],
                        'time': end_time - start_time
                    }
            
            # Show full DAPS results if available
            if 'full_daps_result' in st.session_state:
                res = st.session_state.full_daps_result
                
                col1, col2, col3 = st.columns(3)
                col1.metric("X Coordinate", f"{res['x']:.6f}")
                col2.metric("Y Coordinate", f"{res['y']:.6f}")
                col3.metric("Function Value", f"{res['fun']:.6f}")
                
                st.markdown(f"""
                **Function Evaluations**: {res['nfev']}  
                **Computation Time**: {res['time']:.4f} seconds
                """)
                
                # Compare the demo simulation with the full optimizer
                st.markdown('<div class="result-title">Comparison:</div>', unsafe_allow_html=True)
                
                demo_result = st.session_state.best_result
                full_result = st.session_state.full_daps_result
                
                error = abs(demo_result[2] - full_result['fun'])
                st.markdown(f"""
                Solution difference: {error:.6f}
                
                The difference between the demonstration and the full optimizer is due to:
                1. The demo uses a simplified version of DAPS for visualization purposes.
                2. The full optimizer has additional features like dimensional adaptivity.
                3. The full optimizer may perform more iterations or use different convergence criteria.
                """)
        
        else:
            st.info("Run the DAPS simulation to see optimization results.")
    
    # Code examples
    st.markdown("### Code Example")
    st.markdown("Here's how to use DAPS in your own code:")
    
    st.code("""
    from daps import daps_minimize, DAPSFunction
    
    # Define your function
    def my_function(x, y, z):
        return x**2 + y**2 + z**2
    
    # Create a DAPSFunction instance
    func = DAPSFunction(
        func=my_function,
        name="Custom Function",
        bounds=[-5, 5, -5, 5, -5, 5]
    )
    
    # Run the optimization
    result = daps_minimize(
        func,
        options={
            'maxiter': 50,
            'min_prime_idx': 5,
            'max_prime_idx': 15,
            'verbose': True
        }
    )
    
    # Print the results
    print(f"Optimal solution: {result['x']}")
    print(f"Function value: {result['fun']}")
    """, language="python")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **DAPS: Dimensionally Adaptive Prime Search**  
    GitHub: [https://github.com/sethuiyer/DAPS](https://github.com/sethuiyer/DAPS)
    """)

if __name__ == "__main__":
    main() 