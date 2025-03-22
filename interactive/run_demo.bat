@echo off
REM Script to run the DAPS interactive demo on Windows
REM This simplifies the process of starting the Streamlit app

echo === DAPS Interactive Demo ===
echo Starting the Streamlit application...

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python before running this script.
    exit /b 1
)

REM Check if the directory exists
if not exist "..\daps" (
    echo WARNING: The DAPS package directory was not found.
    echo The demo may not work correctly if DAPS is not installed.
)

REM Check if requirements are installed
echo Checking requirements...
pip list | findstr "streamlit" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing required packages...
    pip install -r requirements.txt
    
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install requirements.
        echo Please run 'pip install -r requirements.txt' manually.
        exit /b 1
    )
)

REM Ensure DAPS package is installed
pip list | findstr "daps" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing DAPS package in development mode...
    pip install -e ..
    
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install DAPS package.
        echo Please run 'pip install -e ..' from the project root manually.
        exit /b 1
    )
)

REM Run the Streamlit app
echo Launching demo...
streamlit run app.py

REM Check if Streamlit exited with an error
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to start the Streamlit app.
    echo Please make sure Streamlit is installed and try running 'streamlit run app.py' manually.
    exit /b 1
) 