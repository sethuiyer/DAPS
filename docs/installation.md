# Installation

DAPS (Dimensionally Adaptive Prime Search) can be installed in several ways depending on your needs.

## Quick Installation

For most users, the simplest method is to install DAPS directly from PyPI:

```bash
pip install daps
```

This will install the latest stable release of DAPS along with all required dependencies.

## Development Installation

For development or to use the latest features, you can install directly from the GitHub repository:

```bash
pip install git+https://github.com/sethuiyer/DAPS.git
```

## From Source

To install DAPS from source:

```bash
# Clone the repository
git clone https://github.com/sethuiyer/DAPS.git
cd DAPS

# Install in development mode
pip install -e .
```

This will install DAPS in development mode, which means changes to the source code will be immediately reflected without requiring reinstallation.

## Dependencies

DAPS requires the following dependencies:

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- Cython ≥ 0.29.24 (for compilation)
- Pydantic ≥ 1.8.0 (for validation)

These will be automatically installed when installing via pip.

## Optional Dependencies

For running benchmarks and tests:

```bash
pip install pytest pytest-benchmark pytest-cov
```

For development:

```bash
pip install tox black flake8 isort
```

## System Requirements

DAPS uses a compiled C++ backend for high performance. The package includes pre-compiled binaries for:

- Linux (x86_64)
- macOS (x86_64, arm64)
- Windows (x86_64)

On other platforms, the package will attempt to compile from source, which requires:

- A C++ compiler supporting C++17
- CMake ≥ 3.12

## Troubleshooting

If you encounter issues during installation:

1. Make sure you have the latest pip:
   ```bash
   pip install --upgrade pip
   ```

2. If the compilation fails, try installing the wheel package:
   ```bash
   pip install wheel
   ```

3. Ensure you have a compatible C++ compiler:
   - GCC ≥ 7 on Linux
   - Clang ≥ 9 on macOS
   - Visual Studio 2017 or newer on Windows

4. If you're still having trouble, please [open an issue](https://github.com/sethuiyer/DAPS/issues) with details about your system and the error message. 