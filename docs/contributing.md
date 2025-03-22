# Contributing to DAPS

Thank you for your interest in contributing to DAPS (Dimensionally Adaptive Prime Search)! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](https://github.com/sethuiyer/DAPS/blob/main/CODE_OF_CONDUCT.md).

## How Can I Contribute?

There are many ways to contribute to DAPS:

1. Reporting bugs
2. Suggesting enhancements
3. Adding new features or improving existing ones
4. Improving documentation
5. Sharing benchmarks and test functions
6. Writing tests

## Development Workflow

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally
   ```bash
   git clone https://github.com/YOUR_USERNAME/DAPS.git
   cd DAPS
   ```
3. Set up the upstream remote
   ```bash
   git remote add upstream https://github.com/sethuiyer/DAPS.git
   ```
4. Create a virtual environment and install development dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

### Development Process

1. Create a new branch for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```
   or
   ```bash
   git checkout -b fix/your-bugfix-name
   ```

2. Make your changes, following the coding standards

3. Run tests to ensure your changes don't break existing functionality
   ```bash
   pytest
   ```

4. Run style checks
   ```bash
   flake8 daps tests
   black --check daps tests
   ```

5. Commit your changes with a descriptive message
   ```bash
   git commit -m "Your descriptive commit message"
   ```

6. Push your changes to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a pull request against the `main` branch of the original repository

## Pull Request Guidelines

1. Ensure your PR addresses a specific issue (or create one if it doesn't exist)
2. Include a clear description of the changes
3. Add tests for any new functionality
4. Update documentation as needed
5. Make sure all tests pass before submitting
6. Keep PRs focused on a single change

## Coding Standards

### Python Code Style

- Follow PEP 8 style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [Flake8](https://flake8.pycqa.org/) for linting
- Add docstrings to all functions, classes, and modules
- Use type hints where appropriate

### C++ Code Style

- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use consistent indentation (4 spaces)
- Keep line length under 100 characters
- Add comments explaining complex logic

## Testing

- Write unit tests for all new functionality
- Use pytest for running tests
- Ensure all tests pass before submitting a PR
- Aim for high test coverage (>80%)

## Documentation

- Update documentation when adding or changing features
- Write clear, concise documentation with examples
- Check documentation builds correctly by running `mkdocs serve`

## Adding New Test Functions

If you'd like to add a new test function to DAPS:

1. Add your function to `daps/functions.py`
2. Include detailed documentation including:
   - Mathematical formula
   - Domain/bounds
   - Global minimum location and value
   - Characteristics (e.g., multimodal, discontinuous)
3. Add tests for your function in `tests/test_functions.py`
4. Create a visualization if possible

Example:

```python
def my_test_function(x, y, z, param=1.0):
    """
    My Test Function
    
    f(x,y,z) = ... (mathematical formula)
    
    Domain: x,y,z ∈ [-10, 10]
    Global minimum: f(x*, y*, z*) = ... at (x*, y*, z*) = (...)
    
    Characteristics:
    - Multimodal with many local minima
    - Continuous and differentiable
    - Sensitive to parameter 'param'
    
    Parameters:
    ----------
    x, y, z : float
        Input coordinates
    param : float, optional
        Controls the scale of the function
        
    Returns:
    -------
    float
        Function value at (x, y, z)
    """
    # Implementation
    return result
```

## Benchmarking

If you'd like to contribute benchmarks:

1. Add your benchmark to `daps/benchmark`
2. Include comparison with other optimization methods
3. Document your benchmark methodology
4. Provide reproducible code and results

## Getting Help

If you have questions or need help, you can:

- Open an issue on GitHub
- Reach out to the maintainers
- Ask in the discussion section

## Release Process

The DAPS project follows [Semantic Versioning](https://semver.org/) for releases.

1. **Major version (X.0.0)**: Incompatible API changes
2. **Minor version (0.X.0)**: Added functionality in a backwards-compatible manner
3. **Patch version (0.0.X)**: Backwards-compatible bug fixes

## Acknowledgments

Contributors will be acknowledged in the project's documentation and README. By contributing, you agree to license your contributions under the same license as the project.

Thank you for contributing to DAPS! Your efforts help make this project better for everyone. 