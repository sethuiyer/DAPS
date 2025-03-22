#!/bin/bash
# Script to run the DAPS interactive demo
# This simplifies the process of starting the Streamlit app

# Print welcome message
echo "=== DAPS Interactive Demo ==="
echo "Starting the Streamlit application..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH."
    echo "Please install Python 3 before running this script."
    exit 1
fi

# Check if the directory exists
if [ ! -d "../daps" ]; then
    echo "WARNING: The DAPS package directory was not found."
    echo "The demo may not work correctly if DAPS is not installed."
fi

# Check if requirements are installed
echo "Checking requirements..."
if ! pip list | grep -q streamlit; then
    echo "Installing required packages..."
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install requirements."
        echo "Please run 'pip install -r requirements.txt' manually."
        exit 1
    fi
fi

# Ensure DAPS package is installed
if ! pip list | grep -q daps; then
    echo "Installing DAPS package in development mode..."
    pip install -e ..
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install DAPS package."
        echo "Please run 'pip install -e ..' from the project root manually."
        exit 1
    fi
fi

# Run the Streamlit app
echo "Launching demo..."
streamlit run app.py

# Check if Streamlit exited with an error
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start the Streamlit app."
    echo "Please make sure Streamlit is installed and try running 'streamlit run app.py' manually."
    exit 1
fi 