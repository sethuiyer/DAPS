# Algorithm Visualizations

This page showcases various visualizations of the DAPS algorithm's behavior and performance characteristics.

## Search Process Visualization

The following animation demonstrates how DAPS explores a 1D function, progressively refining its search:

![DAPS Steps 1D](assets/fig1_pas_steps_1d.png)

*Figure 1: Visualization of DAPS iterations on a 1D function. Notice how the search domain (shown in blue) shrinks around promising regions while grid resolution increases.*

## Prime-Based Grid Sampling

One of the key innovations in DAPS is the use of prime numbers to create efficient sampling grids:

![Prime Grid Sampling](assets/fig4_prime_grid_sampling.png)

*Figure 2: Visualization of prime-based grid sampling across iterations. Each panel shows how different prime numbers create varying grid patterns in 2D, helping avoid aliasing effects.*

## Performance on Discontinuous Functions

DAPS excels at handling discontinuous functions:

![Discontinuous Comparison](assets/fig2_discontinuous_comparison.png)

*Figure 3: Comparison of DAPS (solid blue) performance on discontinuous functions against gradient-based methods (dotted red) and Nelder-Mead (dashed green). The vertical axis shows objective function value, while horizontal axis shows iterations.*

## Recursive Fractal Cliff Valley Function

The RFCV function is a challenging benchmark specifically designed to test optimization algorithms:

![RFCV Visualization](assets/fig3_rfcv_visualization.png)

*Figure 4: Visualization of the Recursive Fractal Cliff Valley Function (RFCV), a challenging 3D benchmark. The function contains multiple discontinuities, fractal-like structures, and deceptive local minima.*

## Comparative Performance

This table compares DAPS with other optimization methods across key metrics:

![Comparison Table](assets/fig5_comparison_table.png)

*Figure 5: Comparison of optimization methods across key performance metrics. Green cells indicate advantages, while yellow/red cells indicate limitations.*

## Prime Grid Resolution Analysis

The following visualization shows how prime grid resolution affects search efficiency:

<div class="mermaid">
  graph LR
    A[Low Primes: 2,3,5,7] -->|"Coarse Exploration"| B[Global Search]
    C[Medium Primes: 11,13,17,19] -->|"Balanced Search"| D[Regional Focus]
    E[High Primes: 23,29,31,37] -->|"Fine Exploration"| F[Local Refinement]
    B --> D --> F
</div>

## Dimensionality Effects

The following chart illustrates how DAPS performance scales with increasing dimensionality:

<div style="border: 1px solid #ddd; padding: 20px; border-radius: 5px; background-color: #f9f9f9; text-align: center;">
  <p><i>Visualization placeholder: Dimensionality vs. Performance Chart</i></p>
  <p>This chart would show how function evaluations and solution quality vary with increasing dimensions (2D through 6D).</p>
</div>

## Interactive 3D Visualization

For a fully interactive experience, visit our [Interactive Demo](usage/interactive-demo.md) page, where you can:

- Select different test functions
- Observe the optimization process in real-time
- Modify algorithm parameters
- Compare with other optimization methods

## Domain Shrinking Strategy

The domain shrinking strategy is a key component of DAPS:

<div class="mermaid">
  flowchart TD
    A[Initial Domain] --> B{Find Best Point}
    B --> C[Calculate New Domain Center]
    C --> D[Apply Shrink Factor]
    D --> E[New Domain]
    E --> F{Convergence?}
    F -->|No| B
    F -->|Yes| G[Final Solution]
</div>

## Creating Your Own Visualizations

You can generate your own visualizations using the `daps.visualization` module:

```python
from daps import daps_minimize, DAPSFunction
from daps.visualization import plot_optimization_path, plot_grid_evolution

# Define your function
def my_function(x, y):
    return x**2 + y**2

# Create a DAPSFunction
func = DAPSFunction(
    func=my_function,
    name="Simple Quadratic",
    bounds=[-5, 5, -5, 5]
)

# Run optimization with history tracking
result = daps_minimize(
    func,
    options={
        'maxiter': 20,
        'track_history': True  # Enable history tracking
    }
)

# Create visualizations
plot_optimization_path(result['history'], func)
plot_grid_evolution(result['history'], func)
```

## Gallery

### Optimization Paths

<div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center">
  <div style="border: 1px solid #ddd; padding: 10px; text-align: center; width: 45%">
    <p><i>Rosenbrock Function</i></p>
  </div>
  <div style="border: 1px solid #ddd; padding: 10px; text-align: center; width: 45%">
    <p><i>Ackley Function</i></p>
  </div>
  <div style="border: 1px solid #ddd; padding: 10px; text-align: center; width: 45%">
    <p><i>Rastrigin Function</i></p>
  </div>
  <div style="border: 1px solid #ddd; padding: 10px; text-align: center; width: 45%">
    <p><i>RFCV Function</i></p>
  </div>
</div>

## Technical Notes

The visualizations on this page were generated using:

- **Matplotlib** for 2D and 3D static plots
- **Seaborn** for statistical visualizations
- **Mermaid.js** for flowcharts and diagrams
- **Custom JavaScript** for interactive elements

For details on the visualization methodology, refer to the [research paper](paper.md) section on visualization techniques. 