import sys
import numpy as np
import zarr
from scipy.ndimage import find_objects


def crop(zarr_container, dataset, out_container=None, out_dataset=None):

    f = zarr.open(zarr_container, "r")

    arr = f[dataset][:]
    offset = f[dataset].attrs["offset"]
    res = f[dataset].attrs["resolution"]

    slices = find_objects(arr > 0)[0]
    new_offset = [offset[i]+(slices[i].start * res[i]) for i in range(3)]

    # write
    if out_container is None:
        out_container = zarr_container

    if out_dataset is None:
        out_dataset = dataset + "_cropped"

    print(f"Writing {out_dataset} to {out_container}")

    out_f = zarr.open(out_container, "a")
    out_f[out_dataset] = arr[slices]
    out_f[out_dataset].attrs["offset"] = new_offset
    out_f[out_dataset].attrs["resolution"] = res


if __name__ == "__main__":

    f = sys.argv[1]
    ds = sys.argv[2]

    crop(f,ds)
