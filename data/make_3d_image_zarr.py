from tqdm import tqdm
import sys
import glob
from tifffile import imread
import numpy as np
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
import zarr


def create_3d_image(
        img_dir,
        out_zarr,
        out_ds,
        out_voxel_size=[1,1,1],
):

    img_paths = sorted(glob.glob(f"{img_dir}/*.tif")) 

    # get total shape
    im = imread(img_paths[0])
    num_z = len(img_paths)
    shape = (num_z,*im.shape)
    print(f"total voxel shape: {shape}")
    voxel_size = Coordinate(out_voxel_size)
    total_roi = Roi((0,0,0), Coordinate(shape) * voxel_size)

    out_image_ds = prepare_ds(
        out_zarr,
        out_ds,
        total_roi,
        voxel_size,
        np.uint8,
        compressor={"id": "blosc"},
    )

    for i, im_path in tqdm(enumerate(img_paths)):

        print("loading", im_path)
        im = imread(im_path)

        if im.dtype != np.uint8: 
            print("making uint8")
            # rescale to uint8 (we then normalize to float during inference)
            im = (im // 256).astype(np.uint8)

        print("writing")
        write_roi = Roi((i,0,0),(1,*shape[1:])) * voxel_size

        out_image_ds[write_roi] = np.expand_dims(im, axis=0)


if __name__ == "__main__":

    img_dir = sys.argv[1] #path/to/dir/containing/tifs
    out_zarr = sys.argv[2] #path/to/output.zarr
    out_ds = sys.argv[3]
    out_z_res = int(sys.argv[4])
    out_yx_res = int(sys.argv[5])
    out_vs = [out_z_res, out_yx_res, out_yx_res]

    create_3d_image(img_dir, out_zarr, out_ds, out_vs)
