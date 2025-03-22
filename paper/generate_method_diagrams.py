#!/usr/bin/env python3
"""
Generate methodology diagrams for DAPS.
This script creates visual explanations of the DAPS algorithm for documentation.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from matplotlib.patches import Rectangle, Arrow, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.path as mpath
import matplotlib.patches as mpatches

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
highlight_color = '#FF5733'  # Coral
secondary_color = '#3498DB'  # Blue
background_color = '#F8F9FA'  # Light gray

def save_figure(fig, filename):
    """Save figure to both paper/figures and docs/assets directories."""
    fig.savefig(f'figures/{filename}', bbox_inches='tight')
    fig.savefig(f'../docs/assets/{filename}', bbox_inches='tight')
    plt.close(fig)

def generate_daps_flowchart():
    """
    Create a flowchart that explains the DAPS algorithm steps.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Create boxes for each step
    boxes = [
        {"label": "Initialize Search Domain", "pos": (5, 13), "width": 4, "height": 1},
        {"label": "Select Prime Number p", "pos": (5, 11.5), "width": 4, "height": 1},
        {"label": "Create p×p×p Grid in Domain", "pos": (5, 10), "width": 4, "height": 1},
        {"label": "Evaluate Function at Grid Points", "pos": (5, 8.5), "width": 4, "height": 1},
        {"label": "Find Best Point", "pos": (5, 7), "width": 4, "height": 1},
        {"label": "Shrink Domain Around Best Point", "pos": (5, 5.5), "width": 4, "height": 1},
        {"label": "Convergence?", "pos": (5, 4), "width": 4, "height": 1, "shape": "diamond"},
        {"label": "Increase Prime Index", "pos": (2, 4), "width": 4, "height": 1},
        {"label": "Return Best Solution", "pos": (5, 2), "width": 4, "height": 1}
    ]
    
    # Draw boxes
    for i, box in enumerate(boxes):
        if box.get("shape") == "diamond":
            # Create diamond shape
            width = box["width"]
            height = box["height"]
            x, y = box["pos"]
            
            diamond = mpath.Path([
                (x, y + height/2),  # top
                (x + width/2, y),  # right
                (x, y - height/2),  # bottom
                (x - width/2, y),  # left
                (x, y + height/2),  # back to top
            ])
            
            patch = mpatches.PathPatch(
                diamond, 
                facecolor=background_color, 
                edgecolor='black', 
                linewidth=2, 
                alpha=0.8
            )
            ax.add_patch(patch)
        else:
            # Create rectangle
            rect = Rectangle(
                (box["pos"][0] - box["width"]/2, box["pos"][1] - box["height"]/2),
                box["width"], 
                box["height"],
                facecolor=background_color,
                edgecolor='black',
                linewidth=2,
                alpha=0.8,
                zorder=1
            )
            ax.add_patch(rect)
        
        # Add label
        ax.text(
            box["pos"][0], 
            box["pos"][1],
            box["label"],
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold',
            zorder=2
        )
    
    # Connect boxes with arrows
    arrows = [
        {"start": (5, 12.5), "end": (5, 12)},  # Initialize → Select Prime
        {"start": (5, 11), "end": (5, 10.5)},  # Select Prime → Create Grid
        {"start": (5, 9.5), "end": (5, 9)},   # Create Grid → Evaluate
        {"start": (5, 8), "end": (5, 7.5)},   # Evaluate → Find Best
        {"start": (5, 6.5), "end": (5, 6)},   # Find Best → Shrink Domain
        {"start": (5, 5), "end": (5, 4.5)},   # Shrink Domain → Convergence
        {"start": (4.5, 4), "end": (3.5, 4)},  # Convergence → Increase Prime (No)
        {"start": (5, 3.5), "end": (5, 2.5)},  # Convergence → Return Best (Yes)
        {"start": (2, 3.5), "end": (2, 11.5), "curved": True},  # Increase Prime → Select Prime
    ]
    
    # Draw arrows
    for arrow in arrows:
        start = arrow["start"]
        end = arrow["end"]
        
        if arrow.get("curved"):
            # Create a curved arrow for the loop back
            arrow_patch = FancyArrowPatch(
                start, 
                end,
                connectionstyle="arc3,rad=-0.5",
                arrowstyle='-|>',
                mutation_scale=20,
                linewidth=2,
                color='black',
                zorder=0
            )
            ax.add_patch(arrow_patch)
        else:
            # Create straight arrow
            arrow_patch = FancyArrowPatch(
                start, 
                end,
                arrowstyle='-|>',
                mutation_scale=20,
                linewidth=2,
                color='black',
                zorder=0
            )
            ax.add_patch(arrow_patch)
    
    # Add Yes/No labels to convergence arrows
    ax.text(4.75, 3.75, "Yes", fontsize=12, ha='center', va='center')
    ax.text(4, 4.2, "No", fontsize=12, ha='center', va='center')
    
    # Add title
    ax.text(5, 14, "DAPS Algorithm Flowchart", fontsize=18, fontweight='bold', ha='center')
    
    save_figure(fig, 'daps_flowchart.png')

def generate_domain_shrinking_diagram():
    """
    Create a visualization of domain shrinking through iterations.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define function
    def test_function(x, y):
        return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)
    
    # Create grid
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = test_function(X, Y)
    
    # Plot contour
    contour = ax.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, ax=ax, label='Function Value')
    
    # Define iterations to show
    iterations = [
        {"domain": [(-5, 5), (-5, 5)], "best_point": (1.5, -2.0), "prime": 5},
        {"domain": [(-1, 4), (-4, 0)], "best_point": (1.2, -1.8), "prime": 7},
        {"domain": [(0, 2.5), (-3, -0.5)], "best_point": (1.0, -1.5), "prime": 11},
        {"domain": [(0.5, 1.5), (-2, -1)], "best_point": (0.9, -1.6), "prime": 13}
    ]
    
    # Different colors for different iterations
    colors = ['white', 'yellow', 'orange', 'red']
    
    # Plot domains
    for i, iteration in enumerate(iterations):
        domain = iteration["domain"]
        best_point = iteration["best_point"]
        prime = iteration["prime"]
        
        # Draw domain rectangle
        rect = Rectangle(
            (domain[0][0], domain[1][0]),
            domain[0][1] - domain[0][0],
            domain[1][1] - domain[1][0],
            fill=False,
            edgecolor=colors[i],
            linewidth=3 - i * 0.5,
            linestyle=['-', '--', '-.', ':'][i],
            label=f'Iteration {i+1} (p={prime})'
        )
        ax.add_patch(rect)
        
        # Mark best point
        ax.scatter(
            best_point[0],
            best_point[1],
            color=colors[i],
            s=150 - i * 20,
            edgecolor='black',
            zorder=10,
            marker='*' if i == len(iterations) - 1 else 'o'
        )
        
        # Add text for best point
        if i == 0:
            ax.annotate(
                f'Best Point {i+1}',
                xy=best_point,
                xytext=(best_point[0] + 1, best_point[1] + 1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10
            )
            
    # Set labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('DAPS Domain Shrinking Through Iterations')
    ax.legend(loc='upper right')
    
    save_figure(fig, 'domain_shrinking.png')

def generate_prime_vs_regular_sampling():
    """
    Create a comparison between prime-based sampling and regular sampling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Regular grid sampling
    regular_sizes = [4, 8, 12]
    
    # Prime-based sampling
    prime_sizes = [3, 7, 11]
    
    # Plot for regular sampling
    ax = axes[0]
    for size in regular_sizes:
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        
        ax.scatter(X, Y, s=500/size, alpha=0.6, label=f'n={size}')
    
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Regular Grid Sampling')
    ax.legend()
    ax.grid(True)
    
    # Plot for prime-based sampling
    ax = axes[1]
    for prime in prime_sizes:
        x = np.linspace(-5, 5, prime)
        y = np.linspace(-5, 5, prime)
        X, Y = np.meshgrid(x, y)
        
        ax.scatter(X, Y, s=500/prime, alpha=0.6, label=f'p={prime}')
    
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Prime-Based Grid Sampling')
    ax.legend()
    ax.grid(True)
    
    # Main title
    fig.suptitle('Regular Grid vs. Prime-Based Sampling', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_figure(fig, 'prime_vs_regular_sampling.png')

def generate_dimensional_adaptivity():
    """
    Create a visualization of dimensional adaptivity in DAPS.
    """
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 3])
    
    # Create axes
    ax1 = fig.add_subplot(gs[0, :])  # Function variation plot
    ax2 = fig.add_subplot(gs[1, 0])  # Initial domain
    ax3 = fig.add_subplot(gs[1, 1])  # Intermediate
    ax4 = fig.add_subplot(gs[1, 2])  # Final domain
    
    # Function with different variability in each dimension
    def test_function(x, y):
        return np.sin(0.5 * x) + np.cos(3 * y) + 0.1 * x**2
    
    # Create data for variation plot
    x = np.linspace(-5, 5, 100)
    y_values = [test_function(xi, 0) for xi in x]
    x_values = [test_function(0, yi) for yi in x]
    
    # Plot variation along each dimension
    ax1.plot(x, y_values, label='Variation along x', color='blue', linewidth=2)
    ax1.plot(x, x_values, label='Variation along y', color='red', linewidth=2)
    ax1.set_title('Function Variation Along Each Dimension')
    ax1.set_xlabel('Dimension Value')
    ax1.set_ylabel('Function Value')
    ax1.legend()
    ax1.grid(True)
    
    # Create data for domain visualization
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = test_function(X, Y)
    
    # Create contour plots for different domains
    domains = [
        [(-5, 5), (-5, 5)],  # Initial
        [(-4, 4), (-2, 2)],  # Intermediate - more shrinking in y
        [(-3, 3), (-0.5, 0.5)]  # Final - much more shrinking in y
    ]
    
    axes = [ax2, ax3, ax4]
    titles = ["Initial Domain", "Intermediate Domain", "Final Domain"]
    
    for i, (ax, domain, title) in enumerate(zip(axes, domains, titles)):
        # Plot contour
        contour = ax.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.7)
        
        # Draw domain rectangle
        rect = Rectangle(
            (domain[0][0], domain[1][0]),
            domain[0][1] - domain[0][0],
            domain[1][1] - domain[1][0],
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        ax.add_patch(rect)
        
        # Set limits to full domain for visualization
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-5.5, 5.5)
        
        # Add annotations
        if i > 0:
            # Show different shrinking rates
            ax.annotate(
                f'X shrink: {100*(1-(domain[0][1]-domain[0][0])/10):.0f}%',
                xy=(-5, -5),
                xytext=(-5, -4.5),
                fontsize=10,
                color='blue'
            )
            ax.annotate(
                f'Y shrink: {100*(1-(domain[1][1]-domain[1][0])/10):.0f}%',
                xy=(-5, -5),
                xytext=(-5, -5),
                fontsize=10,
                color='red'
            )
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
    
    plt.tight_layout()
    save_figure(fig, 'dimensional_adaptivity.png')

def generate_all_diagrams():
    """Generate all methodology diagrams."""
    print("Generating DAPS flowchart...")
    generate_daps_flowchart()
    
    print("Generating domain shrinking diagram...")
    generate_domain_shrinking_diagram()
    
    print("Generating prime vs regular sampling comparison...")
    generate_prime_vs_regular_sampling()
    
    print("Generating dimensional adaptivity visualization...")
    generate_dimensional_adaptivity()
    
    print("All methodology diagrams generated successfully!")

if __name__ == "__main__":
    generate_all_diagrams() 