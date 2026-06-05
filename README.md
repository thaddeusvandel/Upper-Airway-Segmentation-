# Nasal Airway Segmentation

A deep learning-based system for 3D nasal airway segmentation from medical imaging data. This project uses MONAI's UNet architecture to perform automated segmentation of nasal airways from NRRD format medical images, with comprehensive post-processing including 3D mesh reconstruction, visualization, and quantitative metrics.

## Features

- **3D UNet Segmentation**: State-of-the-art deep learning model for volumetric segmentation
- **Batch Processing**: Process multiple cases efficiently
- **3D Mesh Reconstruction**: Generate STL meshes from segmentation masks
- **Comprehensive Visualization**: 2D slices, 3D reconstructions, and overlay plots
- **Quantitative Metrics**: Dice score, IoU, sensitivity, and precision calculations
- **Cross-platform**: Works on Windows, Linux, and macOS
- **Medical Imaging Support**: Native NRRD format support with SimpleITK

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for inference)

### Setup

1. Clone or download the project:
   ```bash
   git clone <repository-url>
   cd nasalseg
   ```

2. Create a virtual environment:
   ```bash
   python -m venv nasa
   # On Windows:
   nasa\Scripts\activate
   # On Linux/macOS:
   source nasa/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the setup script to create directory structure:
   ```bash
   python setup.py
   ```

## Project Structure

```
nasalseg/
├── configs/                 # Configuration files
│   └── model_config.yaml   # Model and preprocessing settings
├── data/                   # Data directory
│   ├── raw/               # Raw NRRD input files
│   └── processed/         # Preprocessed data (if needed)
├── models/                 # Trained model checkpoints
│   ├── model1_without_P001/
│   ├── model2_with_P001/
│   └── New/
├── outputs/                # Output directory
│   ├── meshes/           # STL mesh files
│   ├── metrics/          # JSON metrics files
│   ├── reconstructions/  # 3D reconstruction images
│   └── visualizations/   # 2D comparison plots
├── src/                   # Source code
│   ├── __init__.py
│   ├── predictor.py      # Main prediction class
│   ├── metrics.py        # Evaluation metrics
│   ├── transforms.py     # Data preprocessing transforms
│   ├── visualization.py  # Plotting and visualization
│   └── mesh_utils.py     # 3D mesh processing
├── tests/                 # Unit tests
├── main.py               # Main entry point
├── setup.py              # Project setup script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

### Single Case Inference

Process a single nasal airway case:

```bash
python main.py --image data/raw/P001_img.nrrd --label data/raw/P001_seg.nrrd
```

Run inference without ground truth labels:

```bash
python main.py --image data/raw/P001_img.nrrd --no-label
```

### Batch Processing

Process multiple cases from a directory:

```bash
python main.py --batch --image-dir data/raw --output-dir outputs/batch_results
```

### Advanced Options

#### Model Selection
Use multiple models for ensemble prediction:

```bash
python main.py --image data/raw/P001_img.nrrd \
               --models models/model1/best_metric_model.pth \
                        models/model2/best_metric_model.pth \
               --model-names "Model 1" "Model 2"
```

#### Output Control
Skip certain outputs to speed up processing:

```bash
python main.py --image data/raw/P001_img.nrrd \
               --no-viz --no-mesh --no-metrics
```

#### Device Selection
Force CPU usage (slower but works without GPU):

```bash
python main.py --image data/raw/P001_img.nrrd --device cpu
```

## Configuration

The `configs/model_config.yaml` file contains all model and preprocessing settings:

- **Model Architecture**: UNet with configurable channels and depths
- **Preprocessing**: Spatial padding, intensity normalization, orientation correction
- **Inference**: Device selection, batch size, threshold settings
- **Output**: Visualization and mesh export options

## Data Format

### Input
- **Images**: NRRD format (`*_img.nrrd`) - CT scans of nasal region
- **Labels**: NRRD format (`*_seg.nrrd`) - Ground truth segmentation masks (optional)

### Output
- **Predictions**: NRRD format segmentation masks
- **Meshes**: STL format 3D surface meshes
- **Visualizations**: PNG format comparison plots and 3D reconstructions
- **Metrics**: JSON format quantitative evaluation results

## Model Training

The provided models were trained on nasal airway CT datasets using MONAI's training pipeline. Two model variants are included:

1. **Model 1 (Without P001)**: Trained excluding patient P001 from the training set
2. **Model 2 (With P001)**: Trained including patient P001 in the training set

## Dependencies

Core dependencies include:

- **PyTorch**: Deep learning framework
- **MONAI**: Medical imaging deep learning library
- **SimpleITK**: Medical image processing
- **NumPy**: Numerical computing
- **Matplotlib**: Visualization
- **scikit-image**: Image processing
- **trimesh**: 3D mesh processing
- **PyYAML**: Configuration file parsing

See `requirements.txt` for complete dependency list.

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information]
```

## Support

For issues and questions:

1. Check the existing issues on GitHub
2. Create a new issue with detailed information
3. Include error messages, your system configuration, and steps to reproduce

## Changelog

### Version 1.0.0
- Initial release
- 3D UNet segmentation
- Batch processing support
- 3D mesh reconstruction
- Comprehensive visualization
- Quantitative metrics calculation</content>
<parameter name="filePath">c:\Users\wilfr\Desktop\Wilfred\Projects\Nasalseg\README.md