"""
Setup script for Nasal Airway Segmentation Inference System
Cross-platform (Windows, Linux, macOS)

Usage:
    python setup_project.py
"""

import os
from pathlib import Path


def create_directory_structure():
    """Create the complete project directory structure."""
    
    print("=" * 60)
    print("Setting up Nasal Airway Segmentation Project")
    print("=" * 60)
    print()
    
    # Define directory structure
    directories = {
        "Source Code": [
            "src",
        ],
        "Models": [
            "models/model1_without_P001",
            "models/model2_with_P001",
        ],
        "Data": [
            "data/raw",
            "data/processed",
        ],
        "Outputs": [
            "outputs/visualizations",
            "outputs/reconstructions",
            "outputs/meshes",
            "outputs/metrics",
        ],
        "Configuration": [
            "configs",
        ],
        "Tests": [
            "tests",
        ],
        "Documentation": [
            "docs",
        ],
    }
    
    # Create directories
    print("Creating directories...")
    for category, paths in directories.items():
        print(f"\n{category}:")
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {path}/")
    
    # Create __init__.py for src package
    (Path("src") / "__init__.py").touch()
    
    # Create .gitkeep files to preserve empty directories
    gitkeep_dirs = [
        "data/raw",
        "data/processed",
        "outputs/visualizations",
        "outputs/reconstructions",
        "outputs/meshes",
        "outputs/metrics",
        "tests",
        "docs",
    ]
    
    for directory in gitkeep_dirs:
        (Path(directory) / ".gitkeep").touch()
    
    print()
    return True


def create_gitignore():
    """Create .gitignore file."""
    print("Creating .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt
*.ckpt
!models/**/*.pth  # Keep model weights

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data
data/processed/*
!data/processed/.gitkeep
outputs/*
!outputs/.gitkeep

# OS
.DS_Store
Thumbs.db

# Temporary files
*.log
*.tmp
temp/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("  ✓ .gitignore created")
    print()


def create_config():
    """Create model configuration file."""
    print("Creating configs/model_config.yaml...")
    
    config_content = """# Model Configuration for Nasal Airway Segmentation

# Model Architecture
model:
  type: "UNet"
  spatial_dims: 3
  in_channels: 1
  out_channels: 1
  channels: [16, 32, 64, 128, 256]
  strides: [2, 2, 2, 2]
  num_res_units: 2
  dropout: 0.1
  norm: "batch"

# Model Checkpoints
checkpoints:
  model1:
    path: "models/model1_without_P001/best_metric_model.pth"
    name: "Model 1 (Without P001)"
    description: "Trained without P001 in training set"
  model2:
    path: "models/model2_with_P001/best_metric_model.pth"
    name: "Model 2 (With P001)"
    description: "Trained with P001 in training set"

# Preprocessing
preprocessing:
  spatial_size: [128, 128, 128]
  spacing: [1.0, 1.0, 1.0]
  intensity_range:
    a_min: -1000  # HU
    a_max: 500
    b_min: 0.0
    b_max: 1.0
  orientation: "RAS"

# Inference
inference:
  device: "cuda"  # or "cpu"
  batch_size: 1
  threshold: 0.5

# Output
output:
  save_visualizations: true
  save_meshes: true
  save_metrics: true
  visualization_dpi: 300
"""
    
    with open("configs/model_config.yaml", "w") as f:
        f.write(config_content)
    
    print("  ✓ model_config.yaml created")
    print()


def create_readme_instructions():
    """Create a simple INSTRUCTIONS.txt file."""
    print("Creating INSTRUCTIONS.txt...")
    
    instructions = """NASAL AIRWAY SEGMENTATION - SETUP INSTRUCTIONS
===============================================

Project structure has been created!

NEXT STEPS:
-----------

1. DOWNLOAD MODEL WEIGHTS
   - Download your trained model from run_20260105_234234
     → Save as: models/model1_without_P001/best_metric_model.pth
   
   - Download your trained model from run_20251207_032912
     → Save as: models/model2_with_P001/best_metric_model.pth

2. ADD TEST DATA
   - Copy your NRRD image files to: data/raw/
   - Example: data/raw/P001_img.nrrd
   - If you have labels: data/raw/P001_seg.nrrd

3. INSTALL DEPENDENCIES
   Open terminal/command prompt in this directory and run:
   
   pip install -r requirements.txt

4. RUN INFERENCE
   Single case:
   python main.py --image data/raw/P001_img.nrrd --label data/raw/P001_seg.nrrd
   
   Without label:
   python main.py --image data/raw/P001_img.nrrd --no-label
   
   Batch processing:
   python main.py --batch --image-dir data/raw --output-dir outputs

5. VIEW RESULTS
   Results will be saved in outputs/ directory:
   - outputs/visualizations/  (2D comparison images)
   - outputs/reconstructions/ (3D mesh visualizations)
   - outputs/meshes/          (STL files for 3D viewing)
   - outputs/metrics/         (JSON metrics files)

DIRECTORY STRUCTURE:
--------------------
nasalseg/
├── main.py                          # Run this for inference
├── requirements.txt                 # Dependencies list
├── src/                            # Source code modules
│   ├── predictor.py                # Main prediction class
│   ├── metrics.py                  # Metrics calculation
│   ├── transforms.py               # Data preprocessing
│   ├── visualization.py            # Plotting functions
│   └── mesh_utils.py               # 3D reconstruction
├── models/                         # YOUR MODEL WEIGHTS GO HERE
│   ├── model1_without_P001/
│   │   └── best_metric_model.pth  ← DOWNLOAD HERE
│   └── model2_with_P001/
│       └── best_metric_model.pth  ← DOWNLOAD HERE
├── data/
│   └── raw/                        # YOUR INPUT DATA GOES HERE
│       ├── P001_img.nrrd          ← PLACE NRRD FILES HERE
│       └── P001_seg.nrrd
├── outputs/                        # Results appear here
└── configs/
    └── model_config.yaml           # Configuration settings

TROUBLESHOOTING:
----------------
- If CUDA error: Add --device cpu to use CPU instead
- If out of memory: Process one case at a time
- If import errors: Ensure all dependencies are installed

For more details, see README.md

"""
    
    with open("INSTRUCTIONS.txt", "w") as f:
        f.write(instructions)
    
    print("  ✓ INSTRUCTIONS.txt created")
    print()


def print_summary():
    """Print setup summary and next steps."""
    print()
    print("=" * 60)
    print(" PROJECT SETUP COMPLETE!")
    print("=" * 60)
    print()
    print(" Project Structure:")
    print()
    print("nasalseg/")
    print("├── src/                    # Source code (empty - add your modules)")
    print("├── models/                 #   DOWNLOAD MODEL WEIGHTS HERE")
    print("│   ├── model1_without_P001/")
    print("│   └── model2_with_P001/")
    print("├── data/raw/               # PLACE INPUT NRRD FILES HERE")
    print("├── outputs/                # Results will be saved here")
    print("├── configs/                # Configuration files")
    print("└── tests/                  # Test scripts")
    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print()
    print("1. Download Model Weights:")
    print("   From Google Drive: /content/drive/MyDrive/nasalseg/training_runs/")
    print()
    print("   Model 1 (run_20260105_234234/best_metric_model.pth)")
    print("   → Save to: models/model1_without_P001/best_metric_model.pth")
    print()
    print("   Model 2 (run_20251207_032912/best_metric_model.pth)")
    print("   → Save to: models/model2_with_P001/best_metric_model.pth")
    print()
    print("2. Add Test Data:")
    print("   Copy NRRD files to: data/raw/")
    print("   Example: data/raw/P001_img.nrrd")
    print()
    print("3. Install Dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("4.  Add Source Code:")
    print("   Create the Python modules in src/ directory:")
    print("   - predictor.py")
    print("   - metrics.py")
    print("   - transforms.py")
    print("   - visualization.py")
    print("   - mesh_utils.py")
    print()
    print("5.  Run Inference:")
    print("   python main.py --image data/raw/P001_img.nrrd")
    print()
    print("=" * 60)
    print()
    print(" See INSTRUCTIONS.txt for detailed step-by-step guide")
    print()


def main():
    """Main setup function."""
    # Create directory structure
    create_directory_structure()
    
    # Create configuration files
    create_gitignore()
    create_config()
    create_readme_instructions()
    
    # Print summary
    print_summary()


if __name__ == "__main__":
    main()