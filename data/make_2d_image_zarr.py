import sys
import glob
from tifffile import imread
import numpy as np
import zarr

img_dir = sys.argv[1] #path/to/dir/containing/tifs
out_zarr = sys.argv[2] #path/to/output.zarr

ims = sorted(glob.glob(f"{img_dir}/*.tif")) 
out = zarr.open(out_zarr, "a")

for i, im_path in enumerate(ims):

    print("loading", i)
    im = imread(im_path)

    print("making uint8")
    # rescale to uint8 (we then normalize to float during inference)
    im_uint8 = (im // 256).astype(np.uint8)

    print("writing")

    ds = out.create_dataset(f"image/{i}/s0",data=im_uint8,chunks=(256,256),compressor=zarr.get_codec({'id':"blosc"}),dtype=np.uint8)
    ds.attrs["offset"] = [0] * 2
    ds.attrs["resolution"] = [1, 1]
