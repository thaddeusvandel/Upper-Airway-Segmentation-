"""
3D mesh generation and export utilities.
"""

import numpy as np
from skimage import measure
import trimesh


def create_mesh_from_mask(
    mask: np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
    name: str = "Mesh",
    verbose: bool = True
) -> tuple:
    """
    Create 3D mesh from binary segmentation mask using marching cubes.
    
    Args:
        mask: 3D binary mask (Z, Y, X)
        spacing: Voxel spacing in mm (spacing_x, spacing_y, spacing_z)
        name: Name for logging
        verbose: Whether to print progress
        
    Returns:
        Tuple of (vertices, faces, trimesh_object)
    """
    if verbose:
        print(f"\nGenerating mesh for {name}...")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Spacing: {spacing}")
    
    # Check if mask has any positive voxels
    if mask.sum() == 0:
        if verbose:
            print(f"  Warning: Empty mask for {name}")
        return None, None, None
    
    try:
        # Generate mesh using marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            mask,
            level=0.5,
            spacing=spacing,
            allow_degenerate=False
        )
        
        if verbose:
            print(f"  Vertices: {len(verts):,}")
            print(f"  Faces: {len(faces):,}")
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        # Calculate mesh properties
        if verbose:
            print(f"  Volume: {mesh.volume:.2f} mm³")
            print(f"  Surface Area: {mesh.area:.2f} mm²")
        
        return verts, faces, mesh
        
    except Exception as e:
        if verbose:
            print(f"  Error generating mesh for {name}: {e}")
        return None, None, None


def save_stl_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    verbose: bool = True
) -> bool:
    """
    Save mesh as STL file.
    
    Args:
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (M, 3)
        output_path: Path to save STL file
        verbose: Whether to print progress
        
    Returns:
        True if successful, False otherwise
    """
    if vertices is None or faces is None:
        if verbose:
            print(f"  Cannot save mesh: Invalid data")
        return False
    
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(output_path)
        
        if verbose:
            print(f"  Saved STL: {output_path}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"  Error saving STL to {output_path}: {e}")
        return False