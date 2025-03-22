# DAPS Interactive Demo

This interactive demo showcases the DAPS (Dimensionally Adaptive Prime Search) algorithm using Streamlit. It provides a visual explanation of how the algorithm works, with controls to adjust parameters and visualize the optimization process.

## Features

- **Interactive Parameter Controls**: Adjust prime numbers, iteration count, and domain bounds.
- **Multiple Test Functions**: Experiment with Rosenbrock, Ackley, Sphere, Rastrigin, and Recursive Fractal Cliff Valley functions.
- **Step-by-Step Visualization**: See how the algorithm proceeds through iterations.
- **3D Visualization**: View the function and optimization path in 3D.
- **Results Comparison**: Compare the demo simulation with the full DAPS optimizer.

## Installation

1. First, make sure you have the DAPS package installed:

```bash
# From the root of the repository
pip install -e .
```

2. Install the demo requirements:

```bash
cd interactive
pip install -r requirements.txt
```

## Running the Demo

### Using the Run Scripts (Recommended)

For Linux/Mac:
```bash
cd interactive
./run_demo.sh
```

For Windows:
```
cd interactive
run_demo.bat
```

These scripts will automatically check for dependencies, install them if necessary, and launch the Streamlit app.

### Manual Method

To run the interactive demo manually:

```bash
cd interactive
streamlit run app.py
```

This will start a local Streamlit server and open the demo in your web browser.

## Usage

1. Select a test function from the sidebar.
2. Adjust the domain bounds and algorithm parameters.
3. Click "Run DAPS Simulation" to see the algorithm in action.
4. Explore the different tabs to view 2D iterations, 3D visualization, and optimization results.
5. Optionally, run the full DAPS optimizer and compare the results.

## Online Demo

The interactive demo is also available online at [https://daps-demo.streamlit.app](https://daps-demo.streamlit.app). This allows you to explore the DAPS algorithm without installing anything locally.

## Screenshots

![DAPS Interactive Demo](../docs/assets/daps_demo_screenshot.png)

## How It Works

The demo simulates the DAPS algorithm by:

1. Creating a grid based on the selected prime number.
2. Evaluating the function at all grid points.
3. Finding the point with the minimum value.
4. Shrinking the domain around this point.
5. Moving to the next prime number and repeating.

This visualizes the key innovation of DAPS - using prime-based grids and adaptive domain shrinking to efficiently locate global minima.

## Implementation Details

For developers interested in how the demo is implemented, see the [implementation documentation](../docs/interactive/implementation.md) for details on the code structure, simulation function, and visualization components. 