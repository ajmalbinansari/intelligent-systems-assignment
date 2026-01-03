"""
Setup script to create the proper project structure for the plant disease detection system.
This script creates necessary directories and moves files to their respective locations.
"""

import os
import shutil
import sys
from pathlib import Path

def create_directories():
    """Create the necessary directory structure."""
    directories = [
        'assets',
        'dataset',
        'Jupyter'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_jupyter_notebook():
    """Copy the Jupyter notebook to the Jupyter directory."""
    source = 'plant_disease_detection.ipynb'
    destination = os.path.join('Jupyter', 'plant_disease_detection.ipynb')
    
    if os.path.exists(source):
        shutil.copy(source, destination)
        print(f"Copied {source} to {destination}")
    else:
        print(f"Warning: Could not find {source}")

def create_assets():
    """Create a placeholder logo for the application."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple placeholder logo
        img = Image.new('RGB', (200, 200), color=(53, 133, 79))
        d = ImageDraw.Draw(img)
        
        # Try to add text
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        d.text((35, 90), "Plant Disease\nDetection", fill=(255, 255, 255), font=font)
        
        # Save the logo
        os.makedirs('assets', exist_ok=True)
        img.save('assets/app_logo.png')
        print("Created placeholder logo: assets/app_logo.png")
        
    except ImportError:
        print("Warning: Pillow not installed. Skipping logo creation.")

def create_placeholder_readme():
    """Create a placeholder for the README.md file."""
    if not os.path.exists('README.md'):
        with open('README.md', 'w') as f:
            f.write("# Plant Disease Detection System\n\n")
            f.write("A deep learning system for detecting diseases in plant leaves.\n\n")
            f.write("## Setup\n\n")
            f.write("1. Install requirements: `pip install -r requirements.txt`\n")
            f.write("2. Run the application: `python app.py`\n")
        print("Created placeholder README.md")

def copy_required_files():
    """Make sure all the Python files are in the correct location."""
    # Here we assume that the files are already in the root directory
    # In a real deployment, you might need to copy them from elsewhere
    required_files = [
        'app.py',
        'plant_disease_app.py', 
        'train_model.py',
        'requirements.txt'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Warning: Required file {file} not found.")

def main():
    """Set up the project structure."""
    print("Setting up project structure for Plant Disease Detection system...")
    
    # Create directory structure
    create_directories()
    
    # Set up Jupyter notebook
    setup_jupyter_notebook()
    
    # Create assets
    create_assets()
    
    # Check for required files
    copy_required_files()
    
    # Create placeholder README if needed
    create_placeholder_readme()
    
    print("\nProject structure setup complete!")
    print("\nTo install requirements:")
    print("pip install -r requirements.txt")
    print("\nTo train the model:")
    print("python train_model.py")
    print("\nTo run the application:")
    print("python app.py")

if __name__ == "__main__":
    main()
