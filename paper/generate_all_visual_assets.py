#!/usr/bin/env python3
"""
Generate all visual assets for the DAPS project.
This script runs all figure generation scripts to create a complete set of visual assets.
"""
import os
import sys
import subprocess

def ensure_directories():
    """Ensure all required directories exist."""
    os.makedirs('figures', exist_ok=True)
    os.makedirs('../docs/assets', exist_ok=True)

def run_script(script_name):
    """Run a Python script and report success/failure."""
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Successfully ran {script_name}")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    else:
        print(f"❌ Failed to run {script_name}")
        print(f"   Error: {result.stderr.strip()}")
        return False

def generate_all_assets():
    """Generate all visual assets by running individual scripts."""
    ensure_directories()
    
    # List of scripts to run
    scripts = [
        'generate_logo.py',          # Generate the DAPS logo
        'generate_figures.py',       # Generate paper figures
        'generate_method_diagrams.py'  # Generate methodology diagrams
    ]
    
    # Run each script
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
    
    # Report summary
    print(f"\nGenerated {success_count}/{len(scripts)} visual assets successfully.")
    print(f"Assets are available in 'paper/figures/' and 'docs/assets/' directories.")
    
    # List generated files
    if success_count > 0:
        print("\nGenerated files:")
        for file in sorted(os.listdir('figures')):
            if file.endswith('.png'):
                print(f"- {file}")

if __name__ == "__main__":
    print("=== DAPS Visual Asset Generator ===\n")
    generate_all_assets() 