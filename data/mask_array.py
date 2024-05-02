import sys
import os
import zarr
import numpy as np
from skimage.morphology import remove_small_objects
from funlib.persistence import open_ds, prepare_ds

if __name__ == "__main__":

    in_f = sys.argv[1] #"instance_seg_data.zarr"
    in_ds = sys.argv[2]
    mask_f = sys.argv[3]
    mask_ds = sys.argv[4]
    out_f = sys.argv[5]
    out_ds = sys.argv[6] #in_ds + "_mask" #sys.argv[3] #in_ds.replace("in_arr","object_mask")

    in_arr = open_ds(in_f,in_ds)
    in_roi = in_arr.roi
    vs = in_arr.voxel_size
    
    mask_arr = open_ds(mask_f,mask_ds)
    mask_roi = mask_arr.roi

    roi = in_roi.intersect(mask_roi)

    out_data = in_arr.to_ndarray(roi) * mask_arr.to_ndarray(roi)

    print(f"writing..{out_ds}")
    new_out = prepare_ds(
            out_f,
            out_ds,
            roi,
            vs,
            in_arr.dtype,
            compressor={"id": "blosc", "clevel": 5},
            delete=True)

    new_out[roi] = out_data
