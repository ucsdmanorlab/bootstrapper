import sys
import os
import zarr
import numpy as np
from skimage.morphology import remove_small_objects
from funlib.persistence import open_ds, prepare_ds

if __name__ == "__main__":

    in_f = sys.argv[1] #"instance_seg_data.zarr"
    labels_ds = sys.argv[2]
    unlabelled_ds = labels_ds.replace("ids","mask")

    labels = open_ds(in_f,labels_ds)
    roi = labels.roi
    vs = labels.voxel_size

    labels_arr = labels.to_ndarray(roi)
    unlabelled_arr = labels.to_ndarray(roi) > 0
    unlabelled_arr = (unlabelled_arr).astype(np.uint8)

    print(f"writing..{unlabelled_ds}")
    new_unlabelled = prepare_ds(
            in_f,
            unlabelled_ds,
            roi,
            vs,
            unlabelled_arr.dtype,
            compressor={"id": "blosc", "clevel": 5},
            delete=True)

    new_unlabelled[roi] = unlabelled_arr
