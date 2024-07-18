import sys
import os
import zarr
import numpy as np
from skimage.morphology import binary_closing, disk
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Coordinate, Roi
import sys

if __name__ == "__main__":

    in_f = sys.argv[1] #"instance_seg_data.zarr"
    raw_ds = sys.argv[2]
    mask_ds = sys.argv[3]

    raw = open_ds(in_f,raw_ds)
    roi = raw.roi
    vs = raw.voxel_size

    print(f"loading {raw_ds} from {in_f}")
    mask_arr = raw.to_ndarray(roi) > 0

    print("doing closing")
    footprint = np.stack([
        np.zeros_like(disk(20)),
        disk(20),
        np.zeros_like(disk(20))], axis=0)

    mask_arr = binary_closing(mask_arr, footprint)
    mask_arr = binary_closing(mask_arr, footprint)
    mask_arr = (mask_arr).astype(np.uint8)

    print(f"writing..{mask_ds}")
    new_mask = prepare_ds(
            in_f,
            mask_ds,
            roi,
            vs,
            mask_arr.dtype,
            compressor={"id": "blosc", "clevel": 5},
            write_size=Coordinate((8,256,256))*vs,
            delete=True)

    new_mask[roi] = mask_arr
