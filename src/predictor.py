"""
Nasal Airway Segmentation Predictor
====================================
Main prediction class for 3D upper airway segmentation with reconstruction.

Author: Wilfred Ayine 
Date: January 2026
"""

import torch
import numpy as np
import nrrd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime

from monai.networks.nets import UNet
from monai.transforms import (
    Compose, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd,
    CenterSpatialCropd, Lambdad, EnsureTyped
)

from .metrics import SegmentationMetrics
from .transforms import LoadNrrd, DebugPrintShapeD
from .visualization import (
    create_2d_comparison_plot,
    create_metrics_comparison_plot,
    create_3d_reconstruction_plot,
    create_3d_overlay_plot
)
from .mesh_utils import create_mesh_from_mask, save_stl_mesh


class NasalAirwayPredictor:
    """
    End-to-end predictor for nasal airway segmentation with 3D reconstruction.
    
    Uses padding-based preprocessing to preserve anatomical resolution.
    """
    
    def __init__(
        self,
        model_paths: List[str],
        model_names: Optional[List[str]] = None,
        device: str = 'cuda',
        spatial_size: Tuple[int, int, int] = (192, 240, 64),
        model_config: Optional[Dict] = None,
        verbose: bool = True
    ):
        """
        Initialize the predictor.
        
        Args:
            model_paths: List of paths to model checkpoint files
            model_names: Optional list of names for each model
            device: Device to run inference on
            spatial_size: Target spatial size for padding (default: 192, 240, 64)
            model_config: Model architecture configuration
            verbose: Whether to print progress messages
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.spatial_size = spatial_size
        self.verbose = verbose
        
        # Default model configuration
        self.model_config = model_config or {
            'spatial_dims': 3,
            'in_channels': 1,
            'out_channels': 1,
            'channels': (16, 32, 64, 128, 256),
            'strides': (2, 2, 2, 2),
            'num_res_units': 2,
            'dropout': 0.1,
            'norm': 'batch'
        }
        
        # Setup transforms
        self._setup_transforms()
        
        # Load models
        self.models = []
        self.model_names = model_names or [f"Model_{i+1}" for i in range(len(model_paths))]
        self.checkpoints = []
        
        for path, name in zip(model_paths, self.model_names):
            model, checkpoint = self._load_model(path, name)
            self.models.append(model)
            self.checkpoints.append(checkpoint)
        
        if self.verbose:
            print(f"✓ Loaded {len(self.models)} model(s) on {self.device}")
            print(f"✓ Using padding to spatial size: {self.spatial_size}")
    
    def _setup_transforms(self):
        """Setup preprocessing transforms with padding instead of resizing."""
        self.transforms = Compose([
            LoadNrrd(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), 
                     mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=500, 
                                b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            CenterSpatialCropd(keys=["image", "label"], roi_size=self.spatial_size),
            SpatialPadd(keys=["image", "label"], spatial_size=self.spatial_size, mode="constant"),
            Lambdad(keys=["label"], func=lambda x: (x > 0).astype(np.float32)),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ])
    
    def _load_model(self, model_path: str, name: str) -> Tuple[UNet, Dict]:
        """Load a single model from checkpoint."""
        if self.verbose:
            print(f"\nLoading {name} from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model = UNet(**self.model_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        if self.verbose:
            epoch = checkpoint.get('epoch', 'N/A')
            dice_score = checkpoint.get('dice_score', checkpoint.get('best_metric', 'N/A'))
            print(f"  Epoch: {epoch}")
            if isinstance(dice_score, (int, float)):
                print(f"  Dice Score: {dice_score:.4f}")
        
        return model, checkpoint
    
    def predict(
        self,
        image_path: str,
        label_path: Optional[str] = None,
        output_dir: str = 'outputs',
        save_visualizations: bool = True,
        save_meshes: bool = True,
        save_metrics: bool = True
    ) -> Dict:
        """Run prediction on a single case."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        case_name = Path(image_path).stem.replace('_img', '')
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"PROCESSING: {case_name}")
            print(f"{'='*70}")
        
        # Load and preprocess data
        data_dict = {"image": image_path}
        if label_path:
            data_dict["label"] = label_path
        else:
            img_data, header = nrrd.read(image_path)
            dummy_label = np.zeros(img_data.shape, dtype=np.float32)
            temp_label_path = output_dir / f"{case_name}_dummy_label.nrrd"
            nrrd.write(str(temp_label_path), dummy_label, header)
            data_dict["label"] = str(temp_label_path)
        
        transformed_data = self.transforms(data_dict)
        image = transformed_data["image"].unsqueeze(0).to(self.device)
        label = transformed_data["label"].unsqueeze(0).to(self.device)
        
        # Get spacing info
        _, header = nrrd.read(image_path)
        original_spacing = header.get('space directions', np.eye(3))
        spacing = tuple(np.abs(np.diag(original_spacing)[:3]))
        adjusted_spacing = (1.0, 1.0, 1.0)
        
        # Run inference
        if self.verbose:
            print(f"\nRunning inference...")
            print(f"Input shape: {image.shape}")
        
        predictions = []
        predictions_np = []
        
        with torch.no_grad():
            for model in self.models:
                pred = torch.sigmoid(model(image))
                predictions.append(pred)
                pred_np = (pred.cpu().numpy()[0, 0] > 0.5).astype(np.float32)
                predictions_np.append(pred_np)
        
        # Prepare data for analysis
        image_np = image.cpu().numpy()[0, 0]
        label_np = label.cpu().numpy()[0, 0]
        has_ground_truth = label_path is not None
        
        # Calculate metrics
        metrics_results = {}
        if has_ground_truth:
            if self.verbose:
                print(f"\n{'='*70}")
                print("METRICS EVALUATION")
                print(f"{'='*70}")
            
            for i, (pred, name) in enumerate(zip(predictions, self.model_names)):
                metrics = SegmentationMetrics()
                metrics.update(pred.cpu(), label.cpu())
                metrics_dict = metrics.get_averages()
                metrics_results[name] = metrics_dict
                
                if self.verbose:
                    print(f"\n{name}:")
                    metrics.print_summary()
        
        # 3D Mesh Reconstruction
        if self.verbose:
            print(f"\n{'='*70}")
            print("3D RECONSTRUCTION")
            print(f"{'='*70}")
        
        mesh_data = {}
        
        if has_ground_truth:
            verts_gt, faces_gt, _ = create_mesh_from_mask(
                label_np, adjusted_spacing, "Ground Truth", verbose=self.verbose
            )
            mesh_data['ground_truth'] = {'vertices': verts_gt, 'faces': faces_gt}
        
        for i, (pred_np, name) in enumerate(zip(predictions_np, self.model_names)):
            verts, faces, _ = create_mesh_from_mask(
                pred_np, adjusted_spacing, name, verbose=self.verbose
            )
            mesh_data[name] = {'vertices': verts, 'faces': faces}
        
        # Save outputs
        output_paths = {}
        
        if save_meshes:
            meshes_dir = output_dir / 'meshes'
            meshes_dir.mkdir(exist_ok=True)
            
            if has_ground_truth and mesh_data.get('ground_truth'):
                path = meshes_dir / f"{case_name}_ground_truth.stl"
                save_stl_mesh(
                    mesh_data['ground_truth']['vertices'],
                    mesh_data['ground_truth']['faces'],
                    str(path),
                    verbose=self.verbose
                )
                output_paths['ground_truth_stl'] = str(path)
            
            for name in self.model_names:
                if mesh_data.get(name):
                    path = meshes_dir / f"{case_name}_{name.lower().replace(' ', '_')}.stl"
                    save_stl_mesh(
                        mesh_data[name]['vertices'],
                        mesh_data[name]['faces'],
                        str(path),
                        verbose=self.verbose
                    )
                    output_paths[f'{name}_stl'] = str(path)
        
        if save_visualizations:
            viz_dir = output_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            recon_dir = output_dir / 'reconstructions'
            recon_dir.mkdir(exist_ok=True)
            
            path_2d = viz_dir / f"{case_name}_comparison.png"
            create_2d_comparison_plot(
                image_np, label_np, predictions_np, self.model_names,
                str(path_2d), has_ground_truth
            )
            output_paths['2d_comparison'] = str(path_2d)
            
            if has_ground_truth and metrics_results:
                path_metrics = viz_dir / f"{case_name}_metrics.png"
                create_metrics_comparison_plot(
                    metrics_results, str(path_metrics)
                )
                output_paths['metrics_plot'] = str(path_metrics)
            
            path_3d = recon_dir / f"{case_name}_3d_reconstruction.png"
            create_3d_reconstruction_plot(
                mesh_data, self.model_names, str(path_3d), has_ground_truth
            )
            output_paths['3d_reconstruction'] = str(path_3d)
            
            if has_ground_truth:
                path_overlay = recon_dir / f"{case_name}_3d_overlay.png"
                create_3d_overlay_plot(
                    mesh_data, self.model_names, str(path_overlay)
                )
                output_paths['3d_overlay'] = str(path_overlay)
        
        if save_metrics and metrics_results:
            metrics_dir = output_dir / 'metrics'
            metrics_dir.mkdir(exist_ok=True)
            
            path_json = metrics_dir / f"{case_name}_metrics.json"
            metrics_export = {
                'case_name': case_name,
                'timestamp': datetime.now().isoformat(),
                'models': self.model_names,
                'metrics': metrics_results,
                'spatial_size': list(self.spatial_size),
                'spacing': {
                    'original': list(spacing),
                    'adjusted': list(adjusted_spacing)
                },
                'preprocessing': 'padding (preserves resolution)'
            }
            
            with open(path_json, 'w') as f:
                json.dump(metrics_export, f, indent=2)
            
            output_paths['metrics_json'] = str(path_json)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("✓ PREDICTION COMPLETE")
            print(f"{'='*70}")
            print(f"Outputs saved to: {output_dir}")
        
        return {
            'case_name': case_name,
            'predictions': predictions_np,
            'metrics': metrics_results,
            'mesh_data': mesh_data,
            'output_paths': output_paths,
            'spacing': {'original': spacing, 'adjusted': adjusted_spacing}
        }
    
    def batch_predict(
        self,
        image_paths: List[str],
        label_paths: Optional[List[str]] = None,
        output_dir: str = 'outputs',
        **kwargs
    ) -> List[Dict]:
        """Run prediction on multiple cases."""
        if label_paths is None:
            label_paths = [None] * len(image_paths)
        
        results = []
        for i, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
            case_name = Path(img_path).stem.replace('_img', '')
            case_output_dir = Path(output_dir) / case_name
            
            if self.verbose:
                print(f"\n{'#'*70}")
                print(f"BATCH PROCESSING: {i+1}/{len(image_paths)}")
                print(f"{'#'*70}")
            
            result = self.predict(
                image_path=img_path,
                label_path=lbl_path,
                output_dir=str(case_output_dir),
                **kwargs
            )
            results.append(result)
        
        return results