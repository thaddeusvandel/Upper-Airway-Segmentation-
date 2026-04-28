"""
Nasal Airway Segmentation Package
==================================

A complete package for 3D nasal airway segmentation with:
- Binary segmentation model inference
- Padding-based preprocessing (preserves anatomical resolution)
- Comprehensive metrics evaluation
- 2D/3D visualization
- STL mesh export
"""

from .predictor import NasalAirwayPredictor
from .metrics import SegmentationMetrics
from .transforms import LoadNrrd, DebugPrintShapeD
from .visualization import (
    create_2d_comparison_plot,
    create_metrics_comparison_plot,
    create_3d_reconstruction_plot,
    create_3d_overlay_plot
)
from .mesh_utils import create_mesh_from_mask, save_stl_mesh

__version__ = "1.0.0"
__author__ = "Wilfred Ayine"

__all__ = [
    'NasalAirwayPredictor',
    'SegmentationMetrics',
    'LoadNrrd',
    'DebugPrintShapeD',
    'create_2d_comparison_plot',
    'create_metrics_comparison_plot',
    'create_3d_reconstruction_plot',
    'create_3d_overlay_plot',
    'create_mesh_from_mask',
    'save_stl_mesh',
]