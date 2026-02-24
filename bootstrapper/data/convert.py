import os
import click
import glob
from tqdm import tqdm
import numpy as np
from skimage.io import imread
from scipy.ndimage import find_objects
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
import logging

logging.getLogger().setLevel(logging.INFO)

def read_from(in_path):
    def load_images(paths):
        logging.info(f"Loading {len(paths)} images")
        full_array = np.zeros(
            (len(paths), *imread(paths[0]).shape)
        )  # assume all images have same shape
        for i, im_path in tqdm(enumerate(paths), total=len(paths)):
            im = imread(im_path)
            if len(im.shape) == 3 and im.shape[-1] == 3:
                im = im[..., 0]
            full_array[i] = im
        return full_array

    def load_single_image(path):
        logging.info(f"Loading single image: {path}")
        im = imread(path)
        if len(im.shape) == 4 and im.shape[-1] == 3:
            im = im[..., 0]
        return im

    if os.path.isdir(in_path):  # load all images in directory
        in_paths = sorted(glob.glob(os.path.join(in_path, "*.*")))
        return load_images(in_paths)
    else:  # load single tif file
        return load_single_image(in_path)


@click.command()
@click.option(
    "--in_path",
    "-i",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    required=True,
    help="Path to input 3D image, or directory of 2D images",
    prompt="Enter the path to the input file or directory",
)
@click.option(
    "--out_array",
    "-o",
    type=click.Path(),
    required=True,
    help="Path to output Zarr array",
    prompt="Enter the path to the output Zarr array",
)
@click.option(
    "--dtype",
    "-d",
    type=str,
    default="uint8",
    show_default=True,
    help="Output data type",
)
@click.option(
    "--voxel_size",
    "-vs",
    nargs=3,
    type=int,
    default=(1, 1, 1),
    show_default=True,
    help="Size of each voxel in physical units (space-separated integers)",
)
@click.option(
    "--voxel_offset",
    "-vo",
    nargs=3,
    default=(0, 0, 0),
    type=int,
    help="Offset in voxels (space-separated integers)",
)
@click.option(
    "--axis_names",
    "-ax",
    nargs=3,
    type=str,
    default=("z", "y", "x"),
    show_default=True,
    help="Names of the axes (space-separated strings)",
)
@click.option(
    "--units",
    "-u",
    nargs=3,
    type=str,
    default=("nm", "nm", "nm"),
    show_default=True,
    help="Physical units of the axes (space-separated strings)",
)
@click.option("--crop", "-c", is_flag=True, help="Perform bounding box crop")
def convert(
    in_path, out_array, dtype, voxel_size, voxel_offset, axis_names, units, crop
):
    """Convert a 3D image or directory of 2D images to a Zarr array."""

    # load
    logging.info(
        f"Loading {'directory' if os.path.isdir(in_path) else 'image'}: {in_path}"
    )
    full_array = read_from(in_path)

    shape = full_array.shape
    logging.info(f"Total voxel shape: {shape}, voxel offset: {voxel_offset}")

    # convert dtype
    dtype = np.dtype(dtype)
    min_val, max_val = full_array.min(), full_array.max()
    logging.info(f"Min: {min_val}, Max: {max_val}, array dtype: {full_array.dtype}")

    if dtype not in (np.uint32, np.uint64):
        scale = np.iinfo(dtype).max if 'int' in dtype.name else 1
        scale /= max_val
    else: 
        scale = 1
    logging.info(f"Converting to {dtype}, scaling by {scale}")
    full_array = (scale * full_array).astype(dtype, copy=False)

    # do bounding box crop
    if crop:
        logging.info("Performing bounding box crop")
        bbox = find_objects(full_array > 0)[0]
        full_array = full_array[bbox]
        shape = full_array.shape
        bbox_offset = [x.start for x in bbox]

        logging.info(
            f"Total voxel shape after bounding box: {shape}, voxel_offset: {bbox_offset}"
        )
    else:
        bbox_offset = [0, 0, 0]

    # prepare output dataset
    voxel_size = Coordinate(voxel_size)
    total_roi = (
        Roi(Coordinate(voxel_offset) + Coordinate(bbox_offset), shape) * voxel_size
    )

    if axis_names is not None:
        assert len(axis_names) == len(
            shape
        ), "Number of axis names must match number of dimensions"

    logging.info("Preparing output dataset")
    out_ds = prepare_ds(
        out_array,
        shape=shape,
        offset=total_roi.offset,
        voxel_size=voxel_size,
        dtype=dtype,
        axis_names=axis_names,
        units=units,
    )

    print(f"Writing {out_array}")
    out_ds[total_roi] = full_array

    return out_array, voxel_size


if __name__ == "__main__":
    convert()
