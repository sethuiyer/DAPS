#!/usr/bin/env python3
"""
Generate a logo for DAPS (Dimensionally Adaptive Prime Search)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def generate_daps_logo(output_file='figures/daps_logo.png', dpi=300):
    """
    Generate the DAPS logo and save it to a file.
    """
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Custom colormap from indigo to dark purple
    colors = [(0.3, 0.1, 0.6, 1.0), (0.5, 0.0, 0.9, 1.0)]
    cmap = LinearSegmentedColormap.from_list("daps_gradient", colors)
    
    # Generate 3D grid with prime-based sampling
    x = np.linspace(-4, 4, 11)
    y = np.linspace(-4, 4, 7)
    z = np.linspace(-4, 4, 5)
    
    # Draw 3D grid points with size based on z
    for zi, zval in enumerate(z):
        # Scale points larger as they come forward
        s = 100 * (1 + (zi / len(z)))
        alpha = 0.5 + 0.5 * (zi / len(z))
        
        # Grid points
        ax.scatter(x, [3 - zval] * len(x), s=s, alpha=alpha, 
                  color=cmap(zi/len(z)), edgecolor='white', linewidth=1)
    
    # Add search rays to represent the adaptive nature
    for i in range(5):
        angle = i * np.pi / 12
        length = 4 + i * 0.5
        x_end = length * np.cos(angle)
        y_end = length * np.sin(angle)
        
        # Create a tapered line
        verts = [
            (-1, -1),
            (x_end, y_end),
            (x_end + 0.1, y_end + 0.1),
            (-0.9, -0.9),
        ]
        
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        path = Path(verts, codes)
        patch = PathPatch(path, facecolor=cmap(i/5), alpha=0.3, linewidth=0)
        ax.add_patch(patch)
    
    # Add DAPS text
    ax.text(0, -3, 'DAPS', fontsize=72, fontweight='bold', 
            ha='center', va='center', color='#4B0082')
    
    # Add tagline
    ax.text(0, -4.2, 'Dimensionally Adaptive Prime Search', fontsize=18, 
            ha='center', va='center', color='#6A0DAD')
    
    # Save the logo
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"Logo generated and saved to {output_file}")

if __name__ == "__main__":
    import os
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    
    # Generate the logo
    generate_daps_logo()
    
    # Also generate a copy for docs
    if not os.path.exists('../docs/assets'):
        os.makedirs('../docs/assets', exist_ok=True)
    generate_daps_logo('../docs/assets/daps_logo.png') 