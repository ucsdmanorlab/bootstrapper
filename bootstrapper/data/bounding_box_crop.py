import sys
import os
import numpy as np
import zarr
from scipy.ndimage import find_objects
from funlib.persistence import open_ds, prepare_ds


def crop(zarr_container, dataset, out_container=None, out_dataset=None):

    in_ds = open_ds(os.path.join(zarr_container, dataset))
    arr = in_ds[in_ds.roi]

    slices = find_objects(arr > 0)[0]
    new_offset = [in_ds.offset[i]+(slices[i].start * in_ds.voxel_size[i]) for i in range(3)]

    if out_container is None:
        out_container = zarr_container

    if out_dataset is None:
        out_dataset = dataset + "_cropped"

    cropped_arr = arr[slices]

    print(f"Writing {out_dataset} to {out_container}")

    out_ds = prepare_ds(
        os.path.join(out_container, out_dataset),
        shape=cropped_arr.shape,
        offset=new_offset,
        voxel_size=in_ds.voxel_size,
        axis_names=in_ds.axis_names,
        units=in_ds.units,
        dtype=in_ds.dtype,
        chunk_shape=in_ds.chunk_shape
    )
    out_ds[out_ds.roi] = cropped_arr


if __name__ == "__main__":

    in_f = sys.argv[1]
    in_ds = sys.argv[2]

    try:
        out_f = sys.argv[3]
        out_ds = sys.argv[4]    
    except:
        out_f = None
        out_ds = None

    crop(in_f,in_ds,out_f,out_ds)
