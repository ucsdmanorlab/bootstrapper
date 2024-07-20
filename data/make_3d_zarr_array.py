import os
import sys
import glob
from tqdm import tqdm
import numpy as np
from tifffile import imread
from scipy.ndimage import find_objects
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
import zarr

def create_3d_array(img_path, out_zarr, out_ds, out_dtype, out_voxel_size=[1, 1, 1], voxel_offset=[0, 0, 0]):
   
    # load
    print(f"Loading {'directory' if os.path.isdir(img_path) else 'image'}: {img_path}")
    if os.path.isdir(img_path):
        img_paths = sorted(glob.glob(os.path.join(img_path,"*.tif")))
        full_array = np.zeros((len(img_paths), *imread(img_paths[0]).shape))
        for i, im_path in tqdm(enumerate(img_paths), total=len(img_paths)):
            im = imread(im_path)
            if len(im.shape) == 3 and im.shape[-1] == 3:
                im = im[...,0]
            full_array[i] = im
    else:
        full_array = imread(img_path)
        if len(full_array.shape) == 4 and full_array.shape[-1] == 3:
            full_array = full_array[...,0]
    
    shape = full_array.shape
    print(f"Total voxel shape: {shape}, voxel offset: {voxel_offset}")

    # convert dtype
    if out_dtype == np.uint8 and full_array.dtype != np.uint8:
        full_array = (full_array // 256).astype(np.uint8)
    else:
        full_array = full_array.astype(out_dtype)

    # bounding box
    default_bbox = 'y' if out_dtype != np.uint8 else 'n'
    bbox = input(f"\nPerform bounding box crop? (default: '{default_bbox}'): ") or default_bbox
    if bbox.lower().strip() == 'y':
        bbox = find_objects(full_array > 0)[0]
        full_array = full_array[bbox]
        shape = full_array.shape
        bbox_offset = [x.start for x in bbox]
        
        print(f"Total voxel shape after bounding box: {shape}, voxel_offset: {bbox_offset}")
    else:
        bbox_offset = [0,0,0]

    # prepare output dataset
    voxel_size = Coordinate(out_voxel_size)
    total_roi = Roi(Coordinate(voxel_offset) + Coordinate(bbox_offset), shape) * voxel_size

    out_image_ds = prepare_ds(
        out_zarr,
        out_ds,
        total_roi,
        voxel_size,
        out_dtype,
        compressor={"id": "blosc"},
    )

    print(f"Writing {out_ds} to {out_zarr}..")
    out_image_ds[total_roi] = full_array


if __name__ == "__main__":
    tif_dir_or_path = sys.argv[1]
    out_zarr = sys.argv[2]
    out_ds_name = sys.argv[3]
    out_z_res = int(sys.argv[4])
    out_yx_res = int(sys.argv[5])
    out_vs = [int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[5])]
    voxel_offset = [int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])] if len(sys.argv) == 9 else [0, 0, 0]

    out_dtype = np.uint8 if ('image' in out_ds_name or 'raw' in out_ds_name or 'mask' in out_ds_name) else np.uint64

    create_3d_array(tif_dir_or_path, out_zarr, out_ds_name, out_dtype, out_vs, voxel_offset)
