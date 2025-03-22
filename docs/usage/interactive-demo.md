# Interactive DAPS Demo

<div class="admonition tip">
<p class="admonition-title">Try it live</p>
This interactive demo allows you to experiment with the DAPS algorithm on different test functions.
</div>

## Running the Interactive Demo Locally

For the best experience, you can run the interactive demo locally on your machine. The demo is built with Streamlit and provides real-time visualization of the DAPS algorithm.

### Installation

1. Clone the DAPS repository:
   ```bash
   git clone https://github.com/sethuiyer/DAPS.git
   cd DAPS
   ```

2. Install DAPS in development mode:
   ```bash
   pip install -e .
   ```

3. Install the demo requirements:
   ```bash
   cd interactive
   pip install -r requirements.txt
   ```

### Running the Demo

Start the Streamlit app:

```bash
cd interactive
streamlit run app.py
```

This will open the interactive demo in your web browser.

## Online Demo

The interactive demo is also available online at [https://daps-demo.streamlit.app](https://daps-demo.streamlit.app)

## DAPS Algorithm Visualization

<div class="mermaid">
graph TD
    A[Initialize Search Domain] --> B[Select Prime Number p]
    B --> C[Create p×p×p Grid in Domain]
    C --> D[Evaluate Function at Grid Points]
    D --> E[Find Best Point]
    E --> F[Shrink Domain Around Best Point]
    F --> G{Convergence?}
    G -->|No| H[Increase Prime Index]
    H --> B
    G -->|Yes| I[Return Best Solution]
</div>

## Interactive Optimization

<div class="demo-container">
<div id="interactive-demo">
    <!-- Interactive demo will be loaded here by JavaScript -->
    <p>Loading interactive demo...</p>
</div>
</div>

## Step-by-Step Visualization

The DAPS algorithm proceeds through several iterations, refining the search space:

1. **Initial Sampling**: Using a small prime (p=5)
   ![Initial Sampling](../assets/fig1_pas_steps_1d.png)

2. **Grid Refinement**: As iterations progress, the prime value increases
   ![Grid Refinement](../assets/fig4_prime_grid_sampling.png)

3. **Domain Shrinking**: The search domain contracts around promising regions
   ![Domain Shrinking](../assets/domain_shrinking.png)

## Try Different Test Functions

The interactive demo includes several built-in test functions that showcase DAPS capabilities:

- **Recursive Fractal Cliff Valley**: A challenging function with fractal-like structure
- **Rosenbrock 3D**: Classic banana-shaped valley test function
- **Sphere Function**: Simple convex function
- **Ackley Function**: Highly non-convex function with many local minima
- **Rastrigin Function**: Highly multimodal function with many regular local minima

## Interactive Code Example

```python
from daps import daps_minimize, DAPSFunction
import numpy as np

# Define your own 3D function
def my_function(x, y, z):
    return np.sin(x*y) + np.cos(y*z) + x**2 + y**2 + z**2

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
        'tol': 1e-6
    }
)

print(f"Optimal solution: {result['x']}")
print(f"Function value: {result['fun']}")
print(f"Function evaluations: {result['nfev']}")
```

<script>
document.addEventListener('DOMContentLoaded', function() {
    // This would be replaced with actual interactive demo code
    // For now, we'll just display a message
    const demoContainer = document.getElementById('interactive-demo');
    demoContainer.innerHTML = `
        <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px; text-align: center;">
            <h3>Interactive Demo</h3>
            <p>For the full interactive experience, please run the Streamlit app locally or visit the online demo.</p>
            <p>The local demo provides:</p>
            <ul style="text-align: left;">
                <li>Real-time parameter adjustment</li>
                <li>Detailed visualization of each iteration</li>
                <li>3D function visualization with adjustable viewing angles</li>
                <li>Comparison with the full DAPS optimizer</li>
            </ul>
            <a href="https://github.com/sethuiyer/DAPS/tree/main/interactive" target="_blank" style="
                display: inline-block;
                background-color: #4B0082;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                margin-top: 10px;
            ">Get the Interactive Demo</a>
        </div>
    `;
});
</script>

<style>
.demo-container {
    padding: 20px;
    background-color: #f8f9fa;
    border-radius: 5px;
    margin: 20px 0;
}
</style>

## Performance Comparison

When compared to other optimization methods, DAPS shows excellent performance on discontinuous functions:

![Comparison Table](../assets/fig5_comparison_table.png)

## Prime-Based Grid Sampling

One of the key innovations in DAPS is the use of prime numbers for grid sampling:

![Prime vs Regular Sampling](../assets/prime_vs_regular_sampling.png)

## Dimensional Adaptivity

DAPS adapts differently to each dimension based on the function's behavior:

![Dimensional Adaptivity](../assets/dimensional_adaptivity.png)

## Next Steps

Now that you've seen DAPS in action:

- Try the [examples](examples.md) to see more use cases
- Check out the [benchmarks](../benchmarks.md) for performance data
- Read the [research paper](../paper.md) for the theoretical background 