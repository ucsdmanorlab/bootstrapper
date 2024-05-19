import sys
import os
import zarr
import numpy as np
from skimage.morphology import binary_closing, disk
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Coordinate, Roi
from skimage.transform import rescale
import sys

def upscale(array, factor):

    upscaled = rescale(array, factor, order=0);

    return upscaled

if __name__ == "__main__":

    in_f = sys.argv[1] #"instance_seg_data.zarr"
    raw_ds = sys.argv[2]
    mask_ds = sys.argv[3]

    raw = open_ds(in_f,raw_ds)
    roi = raw.roi
    vs = raw.voxel_size

    print("loading")
    raw_arr = raw.to_ndarray(roi)
    mask_arr = raw.to_ndarray(roi) > 0

    print("doing closing")
    footprint = np.stack([
        np.zeros_like(disk(20)),
        disk(20),
        np.zeros_like(disk(20))], axis=0)

    mask_arr = binary_closing(mask_arr, footprint)
    mask_arr = binary_closing(mask_arr, footprint)
    mask_arr = (mask_arr).astype(np.uint8)

    factor = 2**int(raw_ds[-1])
    factor = (1, factor, factor)
    print(f"upscaling by {factor}")
    mask_arr = upscale(mask_arr, factor) 

    print(f"writing..{mask_ds}")
    new_mask = prepare_ds(
            in_f,
            mask_ds,
            roi,
            vs / Coordinate(factor),
            mask_arr.dtype,
            compressor={"id": "blosc", "clevel": 5},
            write_size=Coordinate((8,256,256))*vs,
            delete=True)

    new_mask[roi] = mask_arr
