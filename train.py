import os
import glob
import argparse
import logging
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, EnsureTyped, SpatialPadd,
    RandCropByPosNegLabeld, RandRotated, RandFlipd, RandZoomd,
    RandGaussianNoised, RandAdjustContrastd, RandAffined
)
from monai.data import CacheDataset, decollate_batch
from monai.networks.nets import AttentionUnet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"MONAI version: {monai.__version__}")

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving outputs to {save_dir}")

    # Load Data
    image_paths = sorted(glob.glob(os.path.join(args.data_dir, "images", "*.nrrd")))
    label_paths = sorted(glob.glob(os.path.join(args.data_dir, "labels", "*.nrrd")))
    
    if not image_paths or not label_paths:
        logger.error(f"No .nrrd files found in {args.data_dir}/images or {args.data_dir}/labels")
        return

    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(image_paths, label_paths)]
    train_files, val_files = train_test_split(data_dicts, test_size=0.2, random_state=42)
    
    logger.info(f"Training samples: {len(train_files)}")
    logger.info(f"Validation samples: {len(val_files)}")

    # Define Transforms
    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=500, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        # Ensure images are at least as large as the crop ROI
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode="constant"),
        
        # Patch-based training (crucial for capturing fine details without static cropping)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=2, # Extract 2 patches per volume
            image_key="image",
            image_threshold=0,
        ),
        
        # Rich Augmentations
        RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest")),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.10),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.5, 2.0)),
        
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=500, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Note: No cropping for validation. We use SlidingWindowInference on the full volume.
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])

    # Datasets and Loaders
    logger.info("Creating cached datasets...")
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.workers)

    from monai.data import list_data_collate
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available())

    # Model
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        dropout=0.1,
    ).to(device) # AttentionUnet naturally handles missing details better by focusing feature maps

    # Loss, Optimizer, Metric
    # DiceFocalLoss is highly effective for class imbalance and thin structures
    loss_function = DiceFocalLoss(sigmoid=True, squared_pred=True, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Training Loop
    best_metric = -1
    best_metric_epoch = -1
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            # list_data_collate automatically folds the num_samples patches into the batch dimension
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                logger.info(f"  Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_loss /= step
        logger.info(f"  Average Training Loss: {epoch_loss:.4f}")
        scheduler.step()

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size=4, predictor=model, overlap=0.5)
                    else:
                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size=4, predictor=model, overlap=0.5)
                    
                    val_outputs_binary = (torch.sigmoid(val_outputs) > 0.5).float()
                    dice_metric(y_pred=val_outputs_binary, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                
                logger.info(f"  Validation Dice: {metric:.4f}")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    save_path = os.path.join(save_dir, "best_metric_model.pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'dice_score': best_metric,
                    }, save_path)
                    logger.info(f"  Saved new best model to {save_path}!")

    logger.info(f"Training Complete! Best Metric: {best_metric:.4f} at epoch {best_metric_epoch}")

    # =====================================================================
    # Post-Training Inference Pipeline
    # =====================================================================
    logger.info("Running final inference on first validation sample...")
    from src.predictor import NasalAirwayPredictor
    
    if len(val_files) > 0:
        val_image_path = val_files[0]["image"]
        val_label_path = val_files[0]["label"]
        best_model_path = os.path.join(save_dir, "best_metric_model.pth")
        
        predictor = NasalAirwayPredictor(
            model_paths=[best_model_path],
            model_names=["Retrained_Model"],
            device=str(device),
            spatial_size=roi_size,
            verbose=True,
            invert=True
        )
        
        result = predictor.predict(
            image_path=val_image_path,
            label_path=val_label_path,
            output_dir="retrain_results",
            save_visualizations=True,
            save_meshes=True,
            save_metrics=True
        )
        logger.info(f"Inference complete! Results saved to retrain_results")
    else:
        logger.warning("No validation files found to run inference on.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NasalSeg Model")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing 'images' and 'labels' subdirs")
    parser.add_argument("--output-dir", type=str, default="training_runs", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (multiply by 2 for num_samples from RandCrop)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-interval", type=int, default=2, help="Validation interval (epochs)")
    parser.add_argument("--roi-x", type=int, default=96, help="Patch size X")
    parser.add_argument("--roi-y", type=int, default=96, help="Patch size Y")
    parser.add_argument("--roi-z", type=int, default=96, help="Patch size Z")
    parser.add_argument("--workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--cache-rate", type=float, default=1.0, help="Dataset cache rate (0.0 - 1.0)")
    
    args = parser.parse_args()
    main(args)
