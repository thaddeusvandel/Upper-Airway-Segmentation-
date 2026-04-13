"""
Segmentation metrics for evaluating model performance.
"""

import torch
import numpy as np
from monai.metrics import DiceMetric


def calculate_iou(pred, target, smooth=1e-6):
    """Intersection over Union (IoU) / Jaccard Index"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def calculate_sensitivity(pred, target, smooth=1e-6):
    """Sensitivity / Recall / True Positive Rate"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    true_positive = (pred * target).sum()
    false_negative = ((1 - pred) * target).sum()

    sensitivity = (true_positive + smooth) / (true_positive + false_negative + smooth)
    return sensitivity.item()


def calculate_specificity(pred, target, smooth=1e-6):
    """Specificity / True Negative Rate"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    true_negative = ((1 - pred) * (1 - target)).sum()
    false_positive = (pred * (1 - target)).sum()

    specificity = (true_negative + smooth) / (true_negative + false_positive + smooth)
    return specificity.item()


def calculate_precision(pred, target, smooth=1e-6):
    """Precision / Positive Predictive Value"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    true_positive = (pred * target).sum()
    false_positive = (pred * (1 - target)).sum()

    precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
    return precision.item()


def calculate_f1_score(pred, target, smooth=1e-6):
    """F1 Score (Harmonic mean of precision and recall)"""
    precision = calculate_precision(pred, target, smooth)
    sensitivity = calculate_sensitivity(pred, target, smooth)

    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + smooth)
    return f1


def calculate_volume_similarity(pred, target, smooth=1e-6):
    """Volume Similarity Coefficient"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    pred_vol = pred.sum()
    target_vol = target.sum()

    vs = 1 - abs(pred_vol - target_vol) / (pred_vol + target_vol + smooth)
    return vs.item()


class SegmentationMetrics:
    """Track all segmentation metrics"""

    def __init__(self):
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.dice_scores = []
        self.iou_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []
        self.precision_scores = []
        self.f1_scores = []
        self.volume_similarities = []
        self.losses = []

    def update(self, pred, target, loss=None):
        """Update metrics with new batch"""
        if pred.dim() == 4:
            pred_binary = (pred > 0.5).float()
        else:
            pred_binary = (pred > 0.5).float()

        if target.dim() == 4:
            target_binary = (target > 0.5).float()
        else:
            target_binary = (target > 0.5).float()

        # Calculate metrics
        try:
            dice = self.dice_metric(pred_binary, target_binary).mean().item()
            self.dice_scores.append(dice)
        except:
            pass

        self.iou_scores.append(calculate_iou(pred, target))
        self.sensitivity_scores.append(calculate_sensitivity(pred, target))
        self.specificity_scores.append(calculate_specificity(pred, target))
        self.precision_scores.append(calculate_precision(pred, target))
        self.f1_scores.append(calculate_f1_score(pred, target))
        self.volume_similarities.append(calculate_volume_similarity(pred, target))

        if loss is not None:
            self.losses.append(loss)

    def get_averages(self):
        """Get average of all metrics"""
        return {
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0,
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0,
            'sensitivity': np.mean(self.sensitivity_scores) if self.sensitivity_scores else 0,
            'specificity': np.mean(self.specificity_scores) if self.specificity_scores else 0,
            'precision': np.mean(self.precision_scores) if self.precision_scores else 0,
            'f1': np.mean(self.f1_scores) if self.f1_scores else 0,
            'volume_similarity': np.mean(self.volume_similarities) if self.volume_similarities else 0,
            'loss': np.mean(self.losses) if self.losses else 0
        }

    def print_summary(self, prefix=""):
        """Print metrics summary"""
        metrics = self.get_averages()
        print(f"  Dice Score:        {metrics['dice']:.4f}")
        print(f"  IoU:               {metrics['iou']:.4f}")
        print(f"  Sensitivity:       {metrics['sensitivity']:.4f}")
        print(f"  Specificity:       {metrics['specificity']:.4f}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  F1 Score:          {metrics['f1']:.4f}")
        print(f"  Volume Similarity: {metrics['volume_similarity']:.4f}")