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
    CenterSpatialCropd, Lambdad, EnsureTyped, Invertd
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
        verbose: bool = True,
        invert: bool = True
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
            invert: If True (default), predictions are mapped back to original
                    patient CT geometry after inference using MONAI Invertd.
                    Set to False to keep outputs in model space (192×240×64).
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.spatial_size = spatial_size
        self.verbose = verbose
        self.invert = invert
        
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
            print(f"[SUCCESS] Loaded {len(self.models)} model(s) on {self.device}")
            print(f"[SUCCESS] Using padding to spatial size: {self.spatial_size}")
            if self.invert:
                print(f"[SUCCESS] Inverse transform enabled - predictions will be restored to original patient space")
            else:
                print(f"[WARNING] Inverse transform disabled - outputs will remain in model space ({self.spatial_size})")
    
    def _setup_transforms(self):
        """Setup preprocessing and (optionally) inverse post-processing transforms.

        ``pre_transforms`` is the unchanged forward pipeline used during both
        training and inference — do NOT reorder its steps.

        ``post_transforms`` uses MONAI's Invertd to replay the recorded
        ``applied_operations`` in reverse, restoring the prediction mask to the
        original patient CT geometry (native shape, spacing, and orientation).
        """
        self.pre_transforms = Compose([
            LoadNrrd(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                     mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=500,
                                 b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            CenterSpatialCropd(keys=["image", "label"], roi_size=self.spatial_size),
            SpatialPadd(keys=["image", "label"], spatial_size=self.spatial_size,
                        mode="constant"),
            Lambdad(keys=["label"], func=lambda x: (x > 0).astype(np.float32)),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ])

        # Keep the legacy attribute so any external callers don't break.
        self.transforms = self.pre_transforms

        # Post-processing: undo spatial transforms for the prediction key.
        # nearest_interp=True ensures binary masks stay binary after resampling.
        self.post_transforms = Compose([
            Invertd(
                keys=["pred"],
                transform=self.pre_transforms,
                orig_keys=["image"],
                meta_keys=["pred_meta_dict"],
                orig_meta_keys=["image_meta_dict"],
                meta_key_postfix="meta_dict",
                nearest_interp=True,
                to_tensor=True,
            )
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
        save_metrics: bool = True,
        invert: Optional[bool] = None
    ) -> Dict:
        """
        Run prediction on a single case.

        Args (additional to existing):
            invert: Override the instance-level ``self.invert`` flag for this
                    call only.  ``None`` means use the instance default.
        """
        # Resolve the invert flag for this call
        do_invert = self.invert if invert is None else invert
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
        
        # Run forward preprocessing.  Keep ``transformed_data`` alive — its
        # MetaTensor objects carry the ``applied_operations`` log that Invertd
        # needs to undo each step in reverse order.
        transformed_data = self.pre_transforms(data_dict)
        image = transformed_data["image"].unsqueeze(0).to(self.device)
        label = transformed_data["label"].unsqueeze(0).to(self.device)
        
        # Read original spacing from the NRRD header (used for mesh generation
        # in original patient space after inversion).
        _, header = nrrd.read(image_path)
        space_dirs = header.get('space directions', np.eye(3))
        original_spacing = tuple(np.abs(np.diag(np.array(space_dirs))[:3]))
        model_spacing = (1.0, 1.0, 1.0)  # spacing after Spacingd resampling
        
        # Run inference
        if self.verbose:
            print(f"\nRunning inference...")
            print(f"Input shape: {image.shape}")
        
        predictions = []
        # model-space binary masks — used for Dice/IoU metrics
        predictions_np_model = []

        with torch.no_grad():
            for model in self.models:
                pred = torch.sigmoid(model(image))
                predictions.append(pred)
                pred_np = (pred.cpu().numpy()[0, 0] > 0.5).astype(np.float32)
                predictions_np_model.append(pred_np)

        # Model-space arrays (correct space for metrics)
        image_np_model = image.cpu().numpy()[0, 0]
        label_np_model = label.cpu().numpy()[0, 0]
        has_ground_truth = label_path is not None

        # ------------------------------------------------------------------ #
        # Inverse transform: restore predictions to original patient geometry #
        # ------------------------------------------------------------------ #
        if do_invert:
            if self.verbose:
                print("\nInverting predictions to original patient space...")

            predictions_np_orig = []
            for pred_tensor in predictions:
                # Detach to CPU MetaTensor and share the image's transform log
                pred_meta = pred_tensor[0].cpu()          # shape: (1, H, W, D)
                pred_meta.applied_operations = (
                    transformed_data["image"].applied_operations.copy()
                )
                transformed_data["pred"] = pred_meta
                inverted = self.post_transforms(transformed_data)
                pred_orig = (inverted["pred"].numpy()[0] > 0.5).astype(np.float32)
                predictions_np_orig.append(pred_orig)
                if self.verbose:
                    print(f"  [SUCCESS] Inverted mask shape: {pred_orig.shape} "
                          f"(original spacing: {original_spacing})")

            # Invert label to original space for visualisations
            lbl_meta = transformed_data["label"].cpu()
            lbl_meta.applied_operations = (
                transformed_data["image"].applied_operations.copy()
            )
            transformed_data["pred"] = lbl_meta
            inverted_label = self.post_transforms(transformed_data)
            label_np_orig = (inverted_label["pred"].numpy()[0] > 0.5).astype(np.float32)

            # Invert image to original space for 2-D slice visualisations
            img_meta = transformed_data["image"].cpu()
            img_meta.applied_operations = (
                transformed_data["image"].applied_operations.copy()
            )
            transformed_data["pred"] = img_meta
            inverted_image = self.post_transforms(transformed_data)
            image_np_orig = inverted_image["pred"].numpy()[0]

            # Use original-space data for mesh and visualisation
            predictions_np = predictions_np_orig
            image_np      = image_np_orig
            label_np      = label_np_orig
            mesh_spacing  = original_spacing
        else:
            # Stay in model space — useful for debugging or fast prototyping
            predictions_np = predictions_np_model
            image_np      = image_np_model
            label_np      = label_np_model
            mesh_spacing  = model_spacing
        
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
                label_np, mesh_spacing, "Ground Truth", verbose=self.verbose
            )
            mesh_data['ground_truth'] = {'vertices': verts_gt, 'faces': faces_gt}

        for i, (pred_np, name) in enumerate(zip(predictions_np, self.model_names)):
            verts, faces, _ = create_mesh_from_mask(
                pred_np, mesh_spacing, name, verbose=self.verbose
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
                    'original': list(original_spacing),
                    'model': list(model_spacing)
                },
                'preprocessing': 'padding (preserves resolution)'
            }
            
            with open(path_json, 'w') as f:
                json.dump(metrics_export, f, indent=2)
            
            output_paths['metrics_json'] = str(path_json)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("[SUCCESS] PREDICTION COMPLETE")
            print(f"{'='*70}")
            print(f"Outputs saved to: {output_dir}")
        
        return {
            'case_name': case_name,
            # Original-space predictions (or model-space if invert=False)
            'predictions': predictions_np,
            # Model-space predictions always available for metric validation
            'predictions_model_space': predictions_np_model,
            'metrics': metrics_results,
            'mesh_data': mesh_data,
            'output_paths': output_paths,
            'spacing': {
                'original': original_spacing,
                'model': model_spacing,
                'inverted': do_invert
            }
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