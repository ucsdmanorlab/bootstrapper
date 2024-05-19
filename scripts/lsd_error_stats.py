from funlib.persistence import open_ds
import numpy as np
import sys
import yaml


def compute_stats(array):

    total_voxels = int(np.prod(array.shape))
    num_nonzero_voxels = array[array > 0].size 
    mean = np.mean(array)
    std = np.std(array)
    
    return {
        'mean': mean,
        'std': std,  
        'num_nonzero_voxels': num_nonzero_voxels,
        'total_voxels': total_voxels,
        'nonzero_ratio' : num_nonzero_voxels / total_voxels,
    }
