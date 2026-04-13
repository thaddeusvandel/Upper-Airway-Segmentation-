"""
Visualization utilities for 2D slices and 3D mesh rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Dict, Optional


def create_2d_comparison_plot(
    image: np.ndarray,
    label: np.ndarray,
    predictions: List[np.ndarray],
    model_names: List[str],
    output_path: str,
    has_ground_truth: bool = True
):
    """
    Create 2D slice comparison visualization across 3 anatomical views.
    
    Args:
        image: Input CT image
        label: Ground truth label
        predictions: List of prediction masks
        model_names: Names for each model
        output_path: Where to save the figure
        has_ground_truth: Whether ground truth is available
    """
    # Select middle slices
    slice_ax = image.shape[2] // 2
    slice_cor = image.shape[1] // 2
    slice_sag = image.shape[0] // 2
    
    views = [
        ("Axial", slice_ax, lambda x: x[:, :, slice_ax]),
        ("Coronal", slice_cor, lambda x: x[:, slice_cor, :]),
        ("Sagittal", slice_sag, lambda x: x[slice_sag, :, :])
    ]
    
    n_models = len(predictions)
    n_cols = 2 + n_models + n_models if has_ground_truth else 1 + n_models
    
    fig = plt.figure(figsize=(4 * n_cols, 12))
    
    colors = ['red', 'blue', 'orange', 'purple']
    
    for view_idx, (view_name, slice_num, slicer) in enumerate(views):
        col = 0
        
        # Ground Truth
        if has_ground_truth:
            ax = plt.subplot(3, n_cols, view_idx * n_cols + col + 1)
            ax.imshow(slicer(image), cmap='gray')
            ax.contour(slicer(label), colors='green', linewidths=2)
            ax.set_title(f'{view_name}\nGround Truth', fontsize=10, fontweight='bold')
            ax.axis('off')
            col += 1
        
        # Model Predictions
        for i, (pred, name) in enumerate(zip(predictions, model_names)):
            ax = plt.subplot(3, n_cols, view_idx * n_cols + col + 1)
            ax.imshow(slicer(image), cmap='gray')
            ax.contour(slicer(pred), colors=colors[i % len(colors)], linewidths=2)
            ax.set_title(f'{view_name}\n{name}', fontsize=10)
            ax.axis('off')
            col += 1
        
        # Error Analysis
        if has_ground_truth:
            for i, (pred, name) in enumerate(zip(predictions, model_names)):
                ax = plt.subplot(3, n_cols, view_idx * n_cols + col + 1)
                error = np.zeros((*slicer(label).shape, 3))
                
                tp = (slicer(pred) == 1) & (slicer(label) == 1)
                fp = (slicer(pred) == 1) & (slicer(label) == 0)
                fn = (slicer(pred) == 0) & (slicer(label) == 1)
                
                error[tp] = [0, 1, 0]
                error[fp] = [1, 0, 0]
                error[fn] = [1, 1, 0]
                
                ax.imshow(slicer(image), cmap='gray')
                ax.imshow(error, alpha=0.5)
                ax.set_title(f'{view_name}\n{name} Errors', fontsize=10)
                ax.axis('off')
                col += 1
    
    # Add legend
    if has_ground_truth:
        legend_ax = fig.add_axes([0.4, 0.02, 0.2, 0.03])
        legend_ax.axis('off')
        legend_elements = [
            Patch(facecolor='green', label='True Positive'),
            Patch(facecolor='red', label='False Positive'),
            Patch(facecolor='yellow', label='False Negative')
        ]
        legend_ax.legend(handles=legend_elements, loc='center', ncol=3, 
                        frameon=False, fontsize=11)
    
    plt.suptitle('Model Predictions - 2D Slice Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_comparison_plot(
    metrics_dict: Dict[str, Dict],
    output_path: str
):
    """
    Create bar chart comparing metrics across models.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    metric_names = ['dice', 'iou', 'sensitivity', 'specificity', 
                   'precision', 'f1', 'volume_similarity']
    x = np.arange(len(metric_names))
    width = 0.8 / len(metrics_dict)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        values = [metrics[m] for m in metric_names]
        offset = width * (i - len(metrics_dict)/2 + 0.5)
        
        bars = ax.bar(x + offset, values, width, 
                     label=model_name, 
                     color=colors[i % len(colors)], 
                     alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], 
                       rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_3d_reconstruction_plot(
    mesh_data: Dict,
    model_names: List[str],
    output_path: str,
    has_ground_truth: bool = True
):
    """
    Create 3D mesh visualization for all models.
    
    Args:
        mesh_data: Dictionary with mesh data for each model
        model_names: Names of models
        output_path: Where to save the figure
        has_ground_truth: Whether ground truth mesh exists
    """
    n_plots = len(model_names) + (1 if has_ground_truth else 0)
    fig = plt.figure(figsize=(6 * n_plots, 6))
    
    plot_idx = 1
    colors = ['green', 'red', 'blue', 'orange']
    
    # Ground truth
    if has_ground_truth and 'ground_truth' in mesh_data:
        ax = fig.add_subplot(1, n_plots, plot_idx, projection='3d')
        _plot_3d_mesh(ax, 
                     mesh_data['ground_truth']['vertices'],
                     mesh_data['ground_truth']['faces'],
                     colors[0], 
                     'Ground Truth\n3D Reconstruction',
                     alpha=0.4)
        plot_idx += 1
    
    # Model predictions
    for i, name in enumerate(model_names):
        if name in mesh_data and mesh_data[name]['vertices'] is not None:
            ax = fig.add_subplot(1, n_plots, plot_idx, projection='3d')
            color_idx = (i + 1) if has_ground_truth else i
            _plot_3d_mesh(ax,
                         mesh_data[name]['vertices'],
                         mesh_data[name]['faces'],
                         colors[color_idx % len(colors)],
                         f'{name}\n3D Reconstruction',
                         alpha=0.4)
        plot_idx += 1
    
    plt.suptitle('3D Mesh Reconstructions - Upper Airway Segmentation',
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_3d_overlay_plot(
    mesh_data: Dict,
    model_names: List[str],
    output_path: str
):
    """
    Create 3D overlay comparison plots.
    
    Args:
        mesh_data: Dictionary with mesh data
        model_names: Names of models
        output_path: Where to save the figure
    """
    n_plots = len(model_names) + 1
    fig = plt.figure(figsize=(7 * n_plots, 6))
    
    colors = ['green', 'red', 'blue', 'orange']
    plot_idx = 1
    
    # GT vs each model
    if 'ground_truth' in mesh_data:
        for i, name in enumerate(model_names):
            if name in mesh_data and mesh_data[name]['vertices'] is not None:
                ax = fig.add_subplot(1, n_plots, plot_idx, projection='3d')
                
                # Plot ground truth
                _plot_3d_mesh(ax,
                             mesh_data['ground_truth']['vertices'],
                             mesh_data['ground_truth']['faces'],
                             'green',
                             f'Ground Truth vs {name}',
                             alpha=0.3)
                
                # Overlay prediction
                mesh = Poly3DCollection(
                    mesh_data[name]['vertices'][mesh_data[name]['faces']], 
                    alpha=0.2, 
                    edgecolor='none'
                )
                mesh.set_facecolor(colors[(i + 1) % len(colors)])
                ax.add_collection3d(mesh)
                
                plot_idx += 1
    
    # Models vs each other
    if len(model_names) >= 2:
        ax = fig.add_subplot(1, n_plots, plot_idx, projection='3d')
        title_parts = []
        
        for i, name in enumerate(model_names[:2]):
            if name in mesh_data and mesh_data[name]['vertices'] is not None:
                mesh = Poly3DCollection(
                    mesh_data[name]['vertices'][mesh_data[name]['faces']],
                    alpha=0.3,
                    edgecolor='none'
                )
                color = colors[(i + 1) % len(colors)]
                mesh.set_facecolor(color)
                ax.add_collection3d(mesh)
                title_parts.append(name)
        
        ax.set_title(' vs '.join(title_parts), fontweight='bold', fontsize=12)
        
        if model_names[0] in mesh_data:
            verts = mesh_data[model_names[0]]['vertices']
            ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
            ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
            ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.view_init(elev=20, azim=45)
    
    plt.suptitle('3D Mesh Overlay Comparisons',
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_3d_mesh(
    ax,
    verts: Optional[np.ndarray],
    faces: Optional[np.ndarray],
    color: str,
    title: str,
    alpha: float = 0.3
):
    """Helper function to plot a single 3D mesh."""
    if verts is None or faces is None:
        ax.text(0.5, 0.5, 0.5, "No mesh data", ha='center', va='center')
        ax.set_title(title)
        return
    
    mesh = Poly3DCollection(verts[faces], alpha=alpha, 
                           edgecolor='k', linewidths=0.1)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    
    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.view_init(elev=20, azim=45)