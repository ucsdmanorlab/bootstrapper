import os
import click
import glob
from tqdm import tqdm
import numpy as np
from tifffile import imread
from scipy.ndimage import find_objects
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
import h5py


def read_from(in_path):
    def load_tif_files(paths):
        full_array = np.zeros((len(paths), *imread(paths[0]).shape))  # assume all images have same shape
        for i, im_path in tqdm(enumerate(paths), total=len(paths)):
            im = imread(im_path)
            if len(im.shape) == 3 and im.shape[-1] == 3:
                im = im[..., 0]
            full_array[i] = im
        return full_array

    def load_single_tif(path):
        im = imread(path)
        if len(im.shape) == 4 and im.shape[-1] == 3:
            im = im[..., 0]
        return im

    def load_h5_file(path):
        with h5py.File(path, 'r') as f:

            datasets = []
            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append(name)

            click.echo("Available datasets:")
            f.visititems(collect_datasets)
            for i, dataset in enumerate(datasets):
                click.echo(f"{i}: {dataset}")

            dataset_index = click.prompt("Enter the number of the dataset to load", type=int)
            dataset_name = datasets[dataset_index]
            return f[dataset_name][...]

    if os.path.isdir(in_path):  # load all tif files in directory
        in_paths = sorted(glob.glob(os.path.join(in_path, "*.tif")))
        return load_tif_files(in_paths)
    elif in_path.endswith('.tif'):  # load single tif file
        return load_single_tif(in_path)
    elif in_path.endswith('.h5'):  # load h5 file
        return load_h5_file(in_path)
    else:
        raise ValueError("Unsupported file format. Supported formats are .tif and .h5")

@click.command()
@click.option('--in_path', '-i', type=click.Path(exists=True, dir_okay=True, file_okay=True), required=True, help='Path to input TIFF stack, directory of TIFF files, h5 file')
@click.option('--out_array', '-o', type=click.Path(), required=True, help='Path to output Zarr array')
@click.option('--dtype', '-d', type=str, default="uint8", help='Output data type', show_default=True, prompt="Enter the output data type")
@click.option('--voxel_size', '-vs', type=(int, int, int), default=(1, 1, 1), help='Size of each voxel', show_default=True, prompt="Enter the voxel size")
@click.option('--voxel_offset', '-vo', type=(int, int, int), default=(0, 0, 0), help='Offset of the voxel grid', show_default=True, prompt="Enter the voxel offset")
@click.option('--axis_names', '-ax', type=(str, str, str), default=['z', 'y', 'x'], help='Names of the axes', show_default=True, prompt="Enter the axis names")
@click.option('--units', '-u', type=(str, str, str), default=['nm', 'nm', 'nm'], help='Units of the axes', show_default=True, prompt="Enter the units")
@click.option('--crop', '-c', is_flag=True, help='Perform bounding box crop')
def make_zarr(in_path, out_array, dtype, voxel_size, voxel_offset, axis_names, units, crop):
    """Convert a TIFF stack or directory of TIFF files to a Zarr array."""
   
    # load
    click.echo(f"Loading {'directory' if os.path.isdir(in_path) else 'image'}: {in_path}")
    full_array = read_from(in_path)
    
    shape = full_array.shape
    click.echo(f"Total voxel shape: {shape}, voxel offset: {voxel_offset}")

    # convert dtype
    dtype = np.dtype(dtype)
    if dtype == np.uint8 and full_array.dtype != np.uint8:
        full_array = (full_array // 256).astype(np.uint8)
    else:
        full_array = full_array.astype(dtype)

    # do bounding box crop
    if crop:
        bbox = find_objects(full_array > 0)[0]
        full_array = full_array[bbox]
        shape = full_array.shape
        bbox_offset = [x.start for x in bbox]
        
        click.echo(f"Total voxel shape after bounding box: {shape}, voxel_offset: {bbox_offset}")
    else:
        bbox_offset = [0,0,0]

    # prepare output dataset
    voxel_size = Coordinate(voxel_size)
    total_roi = Roi(Coordinate(voxel_offset) + Coordinate(bbox_offset), shape) * voxel_size

    if axis_names is not None:
        assert len(axis_names) == len(shape), "Number of axis names must match number of dimensions"

    out_ds = prepare_ds(
        out_array,
        shape=shape,
        offset=total_roi.offset,
        voxel_size=voxel_size,
        dtype=dtype,
        axis_names=axis_names,
        units=units,
    )

    click.echo(f"Writing {out_array}..")
    out_ds[total_roi] = full_array

    return out_array, voxel_size

if __name__ == "__main__":
    make_zarr()