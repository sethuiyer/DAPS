from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy
import os

# Helper function to get absolute paths
def get_path(*args):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *args)

# Define the extension module
ext_modules = [
    Extension(
        "daps.core._daps",  # Module name
        [get_path("daps", "core", "_daps.pyx")],  # Source files
        include_dirs=[numpy.get_include(), get_path("daps", "core")],  # Include NumPy headers
        extra_compile_args=['-std=c++11'],  # Use C++11 standard
        language="c++",
    )
]

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="daps",
    version="0.1.0",
    description="Dimensionally Adaptive Prime Search - High-performance C++/Cython implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sethu Iyer",
    author_email="sethuiyer@gmail.com",
    url="https://github.com/sethuiyer/DAPS",
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    install_requires=[
        "numpy>=1.19.0",
        "pydantic>=1.8.0",
    ],
    extras_require={
        "dev": [
            "cython>=0.29.24",
            "matplotlib>=3.4.0",
            "pytest>=6.0.0",
        ],
    },
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="optimization, 3D optimization, global optimization",
) 