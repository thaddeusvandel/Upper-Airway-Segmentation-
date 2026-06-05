# Automatic Upper Airway Segmentation for CFD Applications

This project focuses on developing a deep learning-based framework for automatic segmentation of the upper airway from medical imaging data. Accurate airway segmentation is essential for computational fluid dynamics (CFD) simulations, which are used to analyze airflow patterns and support the diagnosis of respiratory conditions such as obstructive sleep apnea.

The goal of this work is to create a robust and efficient segmentation pipeline that reduces manual effort while maintaining high anatomical accuracy for downstream CFD analysis.

## Training Pipeline (`train.py`)
To overcome the "Volume vs. Topology" paradox (where models achieve high Dice scores but miss intricate, thin branches critical for CFD), the training pipeline has been heavily optimized:

- **Architecture:** `AttentionUnet` (MONAI)
- **Patch-based Training:** Uses `RandCropByPosNegLabeld` (96x96x96 spatial patches) to prevent strict whole-volume center cropping, ensuring the network learns the full airway topology extending to the outer edges.
- **Loss Function:** `DiceFocalLoss` (gamma=2.0) explicitly penalizes the network for missing thin, difficult-to-segment anatomical structures rather than just measuring bulk volume overlap.
- **Data Augmentations:** `RandAffined`, `RandGaussianNoised`, and `RandAdjustContrastd` ensure the model is robust to varied CT scan qualities and head alignments.
- **Optimization:** Mixed-precision `Adam` optimizer with `CosineAnnealingLR` scheduling.

### Usage
Run the training script (dynamically loads `.nrrd` files from `data/raw`):
```powershell
python train.py --data-dir ./data/raw --epochs 300 --batch-size 2
```

## Inference Pipeline (`main.py`)
Predicts and restores segmentations to the exact spatial resolution and geometry of the source patient CT.
```powershell
python main.py --image data/raw/P001_img.nrrd --label data/raw/P001_seg.nrrd
```
