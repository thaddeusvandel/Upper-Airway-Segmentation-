"""
Data preprocessing transforms for nasal airway segmentation.
"""

import nrrd
import numpy as np
from monai.transforms import MapTransform
from monai.data import MetaTensor


class LoadNrrd(MapTransform):
    """Load NRRD files for medical imaging."""
    
    def __init__(self, keys):
        """
        Args:
            keys: List of keys to load from the data dictionary
        """
        super().__init__(keys)
    
    def __call__(self, data):
        """
        Load NRRD files specified by keys.
        
        Args:
            data: Dictionary with file paths
            
        Returns:
            Dictionary with loaded numpy arrays as MetaTensors
        """
        d = dict(data)
        for key in self.keys:
            filepath = d[key]
            array, header = nrrd.read(filepath)
            
            # Convert to float32 for processing
            array = array.astype(np.float32)
            
            # Create MetaTensor with metadata
            meta_dict = {
                "filename_or_obj": filepath,
                "spatial_shape": array.shape,
                "original_channel_dim": "no_channel",
            }
            
            d[key] = MetaTensor(array, meta=meta_dict)
        return d


class DebugPrintShapeD(MapTransform):
    """Print shape of tensors for debugging."""
    
    def __init__(self, keys, prefix="Debug"):
        """
        Args:
            keys: List of keys to print shapes for
            prefix: Prefix string for print output
        """
        super().__init__(keys)
        self.prefix = prefix
    
    def __call__(self, data):
        """
        Print shapes of specified keys.
        
        Args:
            data: Dictionary with tensor data
            
        Returns:
            Unchanged data dictionary
        """
        d = dict(data)
        for key in self.keys:
            if key in d:
                print(f"{self.prefix} - {key}: shape={d[key].shape}, dtype={d[key].dtype}")
        return d