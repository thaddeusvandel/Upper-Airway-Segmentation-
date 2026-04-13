"""
Main entry point for nasal airway segmentation inference.

Usage:
    python main.py --image data/raw/P001_img.nrrd --label data/raw/P001_seg.nrrd
    python main.py --batch --image-dir data/raw --output-dir outputs/batch_results
    python main.py --image data/raw/P002_img.nrrd --no-label
"""

import argparse
from pathlib import Path
from src.predictor import NasalAirwayPredictor


SCRIPT_DIR = Path(__file__).parent.resolve()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Nasal Airway Segmentation Inference'
    )
    
    # Input arguments
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input NRRD image file'
    )
    parser.add_argument(
        '--label',
        type=str,
        default=None,
        help='Path to ground truth label file (optional)'
    )
    parser.add_argument(
        '--no-label',
        action='store_true',
        help='Run inference without ground truth label'
    )
    
    # Batch processing
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch processing mode'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing input images (for batch mode)'
    )
    parser.add_argument(
        '--label-dir',
        type=str,
        default=None,
        help='Directory containing ground truth labels (for batch mode)'
    )
    
    # Model arguments
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=[
            str(SCRIPT_DIR / 'models' / 'New' / 'best_metric_model.pth')
        ],
        help='Paths to model checkpoint files'
    )
    parser.add_argument(
        '--model-names',
        type=str,
        nargs='+',
        default=['Model 1'],
        help='Names for each model'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--no-mesh',
        action='store_true',
        help='Skip STL mesh export'
    )
    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Skip metrics calculation and export'
    )
    
    # Preprocessing arguments
    parser.add_argument(
        '--spatial-size',
        type=int,
        nargs=3,
        default=[192, 240, 64],
        help='Target spatial size for padding (default: 192 240 64)'
    )
    
    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = NasalAirwayPredictor(
        model_paths=args.models,
        model_names=args.model_names,
        device=args.device,
        spatial_size=tuple(args.spatial_size),
        verbose=not args.quiet
    )
    
    if args.batch:
        # Batch processing mode
        if not args.image_dir:
            raise ValueError("--image-dir required for batch mode")
        
        image_dir = Path(args.image_dir)
        image_paths = sorted(image_dir.glob('*_img.nrrd'))
        
        if not image_paths:
            raise ValueError(f"No *_img.nrrd files found in {image_dir}")
        
        # Find corresponding labels
        label_paths = []
        if args.label_dir and not args.no_label:
            label_dir = Path(args.label_dir)
            for img_path in image_paths:
                case_name = img_path.stem.replace('_img', '')
                label_path = label_dir / f"{case_name}_seg.nrrd"
                label_paths.append(str(label_path) if label_path.exists() else None)
        else:
            label_paths = [None] * len(image_paths)
        
        print(f"\nBatch processing {len(image_paths)} cases...")
        results = predictor.batch_predict(
            image_paths=[str(p) for p in image_paths],
            label_paths=label_paths,
            output_dir=args.output_dir,
            save_visualizations=not args.no_viz,
            save_meshes=not args.no_mesh,
            save_metrics=not args.no_metrics
        )
        
        print(f"\n{'='*70}")
        print(f"Batch processing complete! Processed {len(results)} cases.")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*70}")
        
    else:
        # Single case processing mode
        if not args.image:
            raise ValueError("--image required for single case mode")
        
        label_path = None if args.no_label else args.label
        
        result = predictor.predict(
            image_path=args.image,
            label_path=label_path,
            output_dir=args.output_dir,
            save_visualizations=not args.no_viz,
            save_meshes=not args.no_mesh,
            save_metrics=not args.no_metrics
        )
        
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Case: {result['case_name']}")
        print(f"\nOutput files:")
        for key, path in result['output_paths'].items():
            print(f"  {key}: {path}")
        
        if result['metrics']:
            print(f"\nMetrics:")
            for model_name, metrics in result['metrics'].items():
                print(f"\n  {model_name}:")
                print(f"    Dice:        {metrics['dice']:.4f}")
                print(f"    IoU:         {metrics['iou']:.4f}")
                print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
                print(f"    Precision:   {metrics['precision']:.4f}")
        
        print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
    print("PREDICTION COMPLETE")