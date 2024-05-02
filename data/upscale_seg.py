import zarr
from skimage.transform import rescale
import sys

def upscale(array, factor):

    upscaled = rescale(array, factor, order=0);

    return upscaled

if __name__ == "__main__":

    f_name = sys.argv[1]
    ds = sys.argv[2]
    factor = int(sys.argv[3])
    factor = (1,factor,factor)


    f = zarr.open(f_name,"a")
    arr = f[ds][:]

    voxel_size = f[ds].attrs["resolution"]

    print("writing..")
    f[f"upscaled_{ds}"] = upscale(arr,factor)
    f[f"upscaled_{ds}"].attrs["offset"] = f[ds].attrs["offset"]
    f[f"upscaled_{ds}"].attrs["resolution"] = [int(x/y) for x,y in zip(voxel_size,factor)]
