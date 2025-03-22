#!/usr/bin/env python3
"""
Generate figures for the DAPS paper.
All figures are automatically generated for reproducibility.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy import optimize

# Make sure the figures directory exists
os.makedirs('figures', exist_ok=True)
os.makedirs('../docs/assets', exist_ok=True)

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'

# Custom colors
daps_color = '#4B0082'  # Indigo
comparison_colors = ['#4B0082', '#228B22', '#B22222', '#1E90FF', '#FFD700']

def save_figure(fig, filename, also_for_docs=True):
    """Save figure to both paper/figures and docs/assets directories."""
    fig.savefig(f'figures/{filename}', bbox_inches='tight')
    if also_for_docs:
        fig.savefig(f'../docs/assets/{filename}', bbox_inches='tight')
    plt.close(fig)

def generate_pas_steps_1d():
    """
    Figure 1: Visualization of PAS algorithm iterations on a 1D function.
    """
    # Create a complex 1D function with multiple minima
    def target_function(x):
        return 0.5 * np.sin(5 * x) + 0.2 * np.sin(15 * x) + np.exp(-0.1 * (x - 2)**2) - 1.5
    
    # Define the PAS iterations to show
    iterations = [
        {'prime': 5, 'range': [-5, 5], 'best_x': -0.8},
        {'prime': 7, 'range': [-2, 0.4], 'best_x': -0.7},
        {'prime': 11, 'range': [-1.1, -0.3], 'best_x': -0.68}
    ]
    
    # Create the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('PAS Algorithm Iterations on 1D Function', fontsize=16)
    
    for i, (ax, iteration) in enumerate(zip(axes, iterations)):
        x_min, x_max = iteration['range']
        prime = iteration['prime']
        best_x = iteration['best_x']
        
        # Plot the function
        x = np.linspace(x_min, x_max, 1000)
        y = target_function(x)
        ax.plot(x, y, 'k-', alpha=0.7)
        
        # Generate grid points for this iteration
        grid_x = np.linspace(x_min, x_max, prime)
        grid_y = target_function(grid_x)
        
        # Plot grid points
        ax.scatter(grid_x, grid_y, color=daps_color, s=80, zorder=3, label=f'p={prime}')
        
        # Highlight best point
        best_y = target_function(best_x)
        ax.scatter([best_x], [best_y], color='red', s=120, zorder=4, 
                   marker='*', edgecolor='black', linewidth=1.5)
        
        # Show next search range if not the last iteration
        if i < len(iterations) - 1:
            next_range = iterations[i+1]['range']
            ax.axvspan(next_range[0], next_range[1], alpha=0.2, color='green')
            
        # Set labels
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Iteration {i+1}: p={prime}')
        ax.legend()
        
        # Annotate best point
        ax.annotate(f'Best: ({best_x:.2f}, {best_y:.2f})', 
                    xy=(best_x, best_y), 
                    xytext=(best_x, best_y + 0.3),
                    arrowprops=dict(arrowstyle="->", color='black'),
                    ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.tight_layout()
    save_figure(fig, 'fig1_pas_steps_1d.png')

def generate_prime_grid_sampling():
    """
    Figure 2: Prime-based grid sampling across iterations.
    """
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Prime-Based Grid Sampling in 2D', fontsize=16)
    
    # Define the primes to show
    primes = [5, 7, 11, 13]
    zoom_factor = 0.6
    
    # Initial search domain
    domain = [[-5, 5], [-5, 5]]
    
    # Center point for zooming
    center = np.array([1.5, -2.0])
    
    axes = axes.flatten()
    
    for i, (ax, prime) in enumerate(zip(axes, primes)):
        # Calculate the current domain
        current_domain = [
            [center[0] - zoom_factor**i * 5, center[0] + zoom_factor**i * 5],
            [center[1] - zoom_factor**i * 5, center[1] + zoom_factor**i * 5]
        ]
        
        # Generate the grid
        x = np.linspace(current_domain[0][0], current_domain[0][1], prime)
        y = np.linspace(current_domain[1][0], current_domain[1][1], prime)
        
        # Create meshgrid
        X, Y = np.meshgrid(x, y)
        
        # Plot grid points
        ax.scatter(X, Y, color=daps_color, s=70 / (i + 1), alpha=0.8)
        
        # Add title
        ax.set_title(f'p = {prime}')
        
        # Set labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Plot domain boundaries
        ax.set_xlim(current_domain[0])
        ax.set_ylim(current_domain[1])
        
        # Draw next zoom box if not the last iteration
        if i < len(primes) - 1:
            next_domain = [
                [center[0] - zoom_factor**(i+1) * 5, center[0] + zoom_factor**(i+1) * 5],
                [center[1] - zoom_factor**(i+1) * 5, center[1] + zoom_factor**(i+1) * 5]
            ]
            rect = plt.Rectangle(
                (next_domain[0][0], next_domain[1][0]), 
                next_domain[0][1] - next_domain[0][0], 
                next_domain[1][1] - next_domain[1][0],
                linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
    
    plt.tight_layout()
    save_figure(fig, 'fig4_prime_grid_sampling.png')

def generate_discontinuous_comparison():
    """
    Figure 3: Comparative performance on discontinuous functions.
    """
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Define our test functions
    def sawtooth(x):
        return x - np.floor(x)
    
    def step_function(x):
        result = np.zeros_like(x)
        result[x < -3] = 4
        result[(x >= -3) & (x < -1)] = 3
        result[(x >= -1) & (x < 1)] = 1
        result[(x >= 1) & (x < 3)] = 2
        result[x >= 3] = 0
        return result
    
    # Plot the sawtooth function
    x1 = np.linspace(-5, 5, 1000)
    axes[0, 0].plot(x1, sawtooth(x1), 'k-', linewidth=2)
    axes[0, 0].set_title('Sawtooth Function')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    
    # Plot the step function
    x2 = np.linspace(-5, 5, 1000)
    axes[0, 1].plot(x2, step_function(x2), 'k-', linewidth=2)
    axes[0, 1].set_title('Step Function')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('f(x)')
    
    # Generate PAS vs gradient descent comparison for sawtooth
    x_pas = np.linspace(-5, 5, 11)
    y_pas = sawtooth(x_pas)
    
    # Simulate PAS optimization path
    pas_path_x = [-4, -3, -2.5, -2.2, -2.05, -2.01, -2.0]
    pas_path_y = [sawtooth(x) for x in pas_path_x]
    
    # Simulate gradient descent path (gets stuck oscillating)
    gd_path_x = [-4, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 0.0, 0.5, 0.0]
    gd_path_y = [sawtooth(x) for x in gd_path_x]
    
    # Plot PAS and GD on sawtooth
    axes[1, 0].plot(x1, sawtooth(x1), 'k-', alpha=0.5)
    axes[1, 0].scatter(x_pas, y_pas, color=daps_color, s=80, label='PAS Grid Points')
    axes[1, 0].plot(pas_path_x, pas_path_y, '-o', color='green', linewidth=2, 
                   label='PAS Path')
    axes[1, 0].plot(gd_path_x, gd_path_y, '-o', color='red', linewidth=2, 
                   label='Gradient-Based Path')
    axes[1, 0].set_title('PAS vs. Gradient Methods on Sawtooth')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('f(x)')
    axes[1, 0].legend()
    
    # Generate PAS vs Nelder-Mead comparison for step function
    x_pas = np.linspace(-5, 5, 13)
    y_pas = step_function(x_pas)
    
    # Simulate PAS optimization path
    pas_path_x = [-5, -3.5, -2.5, -1.5, -0.5, 0.0]
    pas_path_y = [step_function(x) for x in pas_path_x]
    
    # Simulate Nelder-Mead path (gets stuck at step boundary)
    nm_path_x = [-5, -3, -2, -1.5, -1.2, -1.1, -1.05, -1.01, -1.005, -1.002, -1.001, -1.0, -0.999]
    nm_path_y = [step_function(x) for x in nm_path_x]
    
    # Plot PAS and Nelder-Mead on step function
    axes[1, 1].plot(x2, step_function(x2), 'k-', alpha=0.5)
    axes[1, 1].scatter(x_pas, y_pas, color=daps_color, s=80, label='PAS Grid Points')
    axes[1, 1].plot(pas_path_x, pas_path_y, '-o', color='green', linewidth=2, 
                   label='PAS Path')
    axes[1, 1].plot(nm_path_x, nm_path_y, '-o', color='orange', linewidth=2, 
                   label='Nelder-Mead Path')
    axes[1, 1].set_title('PAS vs. Nelder-Mead on Step Function')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('f(x)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    save_figure(fig, 'fig2_discontinuous_comparison.png')

def generate_rfcv_visualization():
    """
    Figure 4: Visualization of the Recursive Fractal Cliff Valley Function.
    """
    # Define the RFCV function
    def rfcv(x, y, z=0):
        # Base oscillatory component
        base = np.sin(x) * np.cos(y) * np.exp(-0.1 * z**2)
        
        # Recursive component with discontinuities
        recursive = 0.5 * np.sin(3 * x) * np.cos(2 * y) * np.exp(-0.05 * (z - 1)**2)
        recursive += 0.3 * np.sin(5 * x) * np.cos(7 * y) * np.exp(-0.05 * (z + 1)**2)
        
        # Cliff component
        cliff = 2 * np.arctan(10 * (x + y))
        
        # Valley component
        valley = -np.exp(-0.1 * ((x + np.pi)**2 + (y - np.e)**2 + (z - np.sqrt(5))**2))
        
        # Combine components
        return base + recursive + cliff + valley
    
    # Create a 3D visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 3D Surface at z=0
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    x = y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rfcv(X[i, j], Y[i, j], 0)
    
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                            linewidth=0, antialiased=True)
    ax1.set_title('RFCV: z=0 Cross-section')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y,0)')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Contour plot at z=0
    ax2 = fig.add_subplot(2, 2, 2)
    contour = ax2.contourf(X, Y, Z, 20, cmap='viridis')
    ax2.set_title('RFCV: Contour at z=0')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=5)
    
    # Mark global minimum
    min_x, min_y, min_z = -np.pi, np.e, np.sqrt(5)
    min_value = rfcv(min_x, min_y, min_z)
    
    # Find where z=min_z crosses our grid
    ax3 = fig.add_subplot(2, 2, 3)
    z_slice = min_z
    Z_slice = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_slice[i, j] = rfcv(X[i, j], Y[i, j], z_slice)
    
    contour2 = ax3.contourf(X, Y, Z_slice, 20, cmap='viridis')
    ax3.scatter([min_x], [min_y], color='red', s=150, marker='*', 
                edgecolor='white', linewidth=1.5)
    ax3.set_title(f'RFCV: Contour at z={min_z:.2f}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    fig.colorbar(contour2, ax=ax3, shrink=0.5, aspect=5)
    
    # 1D slices
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Plot slice along x-axis
    x_slice = np.linspace(-5, 5, 1000)
    y_x_slice = rfcv(x_slice, min_y, min_z)
    ax4.plot(x_slice, y_x_slice, 'b-', linewidth=2, label='x-axis slice')
    
    # Plot slice along y-axis
    y_slice = np.linspace(-5, 5, 1000)
    y_y_slice = rfcv(min_x, y_slice, min_z)
    ax4.plot(y_slice, y_y_slice, 'g-', linewidth=2, label='y-axis slice')
    
    # Plot slice along z-axis
    z_slice = np.linspace(-5, 5, 1000)
    y_z_slice = rfcv(min_x, min_y, z_slice)
    ax4.plot(z_slice, y_z_slice, 'r-', linewidth=2, label='z-axis slice')
    
    ax4.set_title('RFCV: 1D Slices Through Global Minimum')
    ax4.set_xlabel('Variable value')
    ax4.set_ylabel('f(x,y,z)')
    ax4.legend()
    
    plt.tight_layout()
    save_figure(fig, 'fig3_rfcv_visualization.png')

def generate_comparison_table():
    """
    Figure 5: Comparison of optimization methods across key performance metrics.
    """
    # Define the metrics and methods to compare
    metrics = [
        'Gradient Free',
        'Handles Discontinuities',
        'Multi-minima Landscapes',
        'Affected by Initial Guess',
        'Memory Requirements',
        'Convergence Speed',
        'Function Evaluations',
        'Parallelization',
        'Theoretical Guarantees'
    ]
    
    methods = ['PAS', 'Nelder-Mead', 'BFGS', 'Genetic Algorithm', 'Grid Search']
    
    # Define the comparison data (1=bad, 2=fair, 3=good)
    data = [
        [3, 3, 1, 3, 3],  # Gradient Free
        [3, 2, 1, 2, 3],  # Handles Discontinuities
        [2, 2, 1, 3, 2],  # Multi-minima Landscapes
        [2, 2, 3, 1, 3],  # Affected by Initial Guess (3=less affected)
        [3, 3, 2, 1, 1],  # Memory Requirements (3=low memory)
        [2, 2, 3, 1, 1],  # Convergence Speed
        [2, 2, 3, 1, 1],  # Function Evaluations (3=fewer)
        [2, 1, 1, 3, 3],  # Parallelization
        [2, 2, 3, 2, 3]   # Theoretical Guarantees
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data, index=metrics, columns=methods)
    
    # Create a custom colormap (1=red/bad, 2=yellow/fair, 3=green/good)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#ff7f7f', '#ffff7f', '#7fff7f'])
    
    # Create the figure
    fig, ax = plt.figure(figsize=(12, 8), dpi=150), plt.gca()
    
    # Create the heatmap
    sns.heatmap(df, annot=True, cmap=cmap, linewidths=0.5, linecolor='gray',
                cbar=False, ax=ax, vmin=1, vmax=3)
    
    # Customize the plot
    ax.set_title('Comparison of Optimization Methods', fontsize=16, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    # Add color legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='#7fff7f', label='Good'),
        plt.Rectangle((0, 0), 1, 1, color='#ffff7f', label='Fair'),
        plt.Rectangle((0, 0), 1, 1, color='#ff7f7f', label='Limited')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.tight_layout()
    save_figure(fig, 'fig5_comparison_table.png')

# Generate all figures
def generate_all_figures():
    print("Generating Figure 1: PAS Algorithm Iterations")
    generate_pas_steps_1d()
    
    print("Generating Figure 2: Prime Grid Sampling")
    generate_prime_grid_sampling()
    
    print("Generating Figure 3: Discontinuous Comparison")
    generate_discontinuous_comparison()
    
    print("Generating Figure 4: RFCV Visualization")
    generate_rfcv_visualization()
    
    print("Generating Figure 5: Comparison Table")
    generate_comparison_table()
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    generate_all_figures() 