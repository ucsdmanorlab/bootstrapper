import sys
import os
import glob
import zarr
import numpy as np
from skimage.io import imread


# This script creates a zarr container containing 2D image crops
# and corresponding ground-truth labels and object masks. It requires
# the images to be .tifs and the labels to be stored in "masks" in a 
# compressed numpy format (.npy), in the same folder. 


def consolidate_data(images, container):
    for idx, f in enumerate(images):
        # get raw image to np array
        im = imread(f)

        # rescale, cast to uint8
        if np.max(im) != 255:
            im = im // 255

        im = im.astype(np.uint8)

        # get labels
        labels = np.load(f.replace(".tif", "_seg.npy"), allow_pickle=True).item()[
            "masks"
        ].astype(np.uint32)

        # make object mask
        object_mask = (labels > 0).astype(np.uint8)
        
        datasets = [("image", im), ("labels", labels), ("object_mask", object_mask)]

        for ds_name, data in datasets:
            ds = f"{ds_name}/{idx}/s0"
            container[ds] = data
            container[ds].attrs["offset"] = [0, 0]
            container[ds].attrs["resolution"] = [1, 1]  # arbitrary for now


if __name__ == "__main__":

    out_container = sys.argv[1]
    img_directories = sys.argv[2:]

    tif_files = []
    for img_dir in img_directories:
        tif_files += sorted(glob.glob(f"{img_dir}/*.tif"))

    instance_seg_container = zarr.open(out_container, "a")

    consolidate_data(tif_files, instance_seg_container)
