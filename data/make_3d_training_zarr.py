from tqdm import tqdm
import os
import sys
import glob
from tifffile import imread
import numpy as np
from funlib.persistence import prepare_ds, open_ds
from funlib.geometry import Coordinate, Roi
import zarr


def create_3d_array_from_image(
        img_path,
        out_zarr,
        out_ds,
        out_dtype,
        out_voxel_size=[1,1,1],
):

    print("loading", img_path)
    im = imread(img_path)
    shape = im.shape
    if shape[-1] == 3:
        shape = shape[:3]
        im = im[:,:,:,0]

    print(f"total voxel shape: {shape}")
    voxel_size = Coordinate(out_voxel_size)
    total_roi = Roi((0,0,0), Coordinate(shape) * voxel_size)

    out_image_ds = prepare_ds(
        out_zarr,
        out_ds,
        total_roi,
        voxel_size,
        out_dtype,
        compressor={"id": "blosc"},
    )
    
    if out_dtype == np.uint8 and im.dtype != np.uint8:
        print("making uint8")
        # rescale to uint8 (we then normalize to float during inference)
        im = (im // 256).astype(np.uint8)
    else:
        im = im.astype(out_dtype)

    print("writing")
    out_image_ds[total_roi] = im


def create_mask(in_f, in_ds, out_ds):

    labels = open_ds(in_f, in_ds)
    roi = labels.roi
    vs = labels.voxel_size

    labels_arr = labels.to_ndarray(roi)
    unlabelled_arr = labels.to_ndarray(roi) > 0
    unlabelled_arr = (unlabelled_arr).astype(np.uint8)

    print(f"writing..{out_ds}")
    unlabelled = prepare_ds(
            in_f,
            out_ds,
            roi,
            vs,
            unlabelled_arr.dtype,
            compressor={"id": "blosc", "clevel": 5},
            delete=True)

    unlabelled[roi] = unlabelled_arr


def create_3d_array_from_images(
        img_dir,
        out_zarr,
        out_ds,
        out_dtype,
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
        out_dtype,
        compressor={"id": "blosc"},
    )

    for i, im_path in tqdm(enumerate(img_paths)):

        print("loading", im_path)
        im = imread(im_path)

        if out_dtype == np.uint8 and im.dtype != np.uint8:
            print("making uint8")
            # rescale to uint8 (we then normalize to float during inference)
            im = (im // 256).astype(np.uint8)
        else:
            im = im.astype(out_dtype)

        print("writing")
        write_roi = Roi((i,0,0),(1,*shape[1:])) * voxel_size

        out_image_ds[write_roi] = np.expand_dims(im, axis=0)


if __name__ == "__main__":

    img_dir_or_path = sys.argv[1]
    labels_dir_or_path = sys.argv[2]
    out_zarr = sys.argv[3] #path/to/output.zarr
    out_image_ds_name = 'volumes/image/s0'
    out_labels_ds_name = 'volumes/labels/ids/s0'
    out_mask_ds_name = 'volumes/labels/mask/s0'
    out_z_res = int(sys.argv[4])
    out_yx_res = int(sys.argv[5])
    out_vs = [out_z_res, out_yx_res, out_yx_res]

    print("creating image volume")
    if os.path.isdir(img_dir_or_path):
       create_fn = create_3d_array_from_images
    else:
       create_fn = create_3d_array_from_image

    create_fn(
       img_dir_or_path,
       out_zarr,
       out_image_ds_name,
       np.uint8,
       out_vs)

    print("creating labels volume")
    if os.path.isdir(labels_dir_or_path):
       create_fn = create_3d_array_from_images
    else:
       create_fn = create_3d_array_from_image

    create_fn(
       labels_dir_or_path,
       out_zarr,
       out_labels_ds_name,
       np.uint64,
       out_vs)

    print("creating mask volume")
    create_mask(out_zarr, out_labels_ds_name, out_mask_ds_name)
