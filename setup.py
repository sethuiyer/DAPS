from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os
import platform

# Define the extensions
extensions = [
    Extension(
        "daps.core._daps",
        ["daps/core/_daps.pyx", "daps/core/daps.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"] if platform.system() != "Windows" else ["/std:c++11"],
    ),
]

# Read the content of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="daps",
    version="0.1.0",
    author="Sethu Iyer",
    author_email="sethumiyer@gmail.com",
    description="Dimensionally Adaptive Prime Search: A high-performance global optimization algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sethuiyer/DAPS",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.6",
    install_requires=["numpy>=1.19.0"],
    ext_modules=cythonize(extensions, 
                         compiler_directives={
                             "language_level": 3,
                             "boundscheck": False, 
                             "wraparound": False,
                             "initializedcheck": False,
                         },
                         annotate=True),
    include_package_data=True,
) 