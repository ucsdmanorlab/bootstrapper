import click
import os
import re
import daisy
from funlib.persistence import open_ds, prepare_ds
import zarr
from functools import partial


def scale_block(in_array, out_array, factor, mode, block):
    import numpy as np
    from funlib.persistence import Array
    from skimage.measure import block_reduce
    from skimage.transform import rescale

    dims = len(factor)
    in_data = in_array.to_ndarray(block.read_roi, fill_value=0)
    name = in_array.data.name
    
    n_channels = len(in_data.shape) - dims
    if n_channels >= 1:
        factor = (1,)*n_channels + factor

    if in_data.dtype in [np.uint32, np.uint64] or 'label' in name or 'id' in name or 'mask' in name:
        if mode == 'down':
            slices = tuple(slice(k//2, None, k) for k in factor)
            out_data = in_data[slices]
        else:  # upscale
            out_data = in_data
            for axis, f in enumerate(factor):
                out_data = np.repeat(out_data, f, axis=axis)
    else:
        if mode == 'down':
            out_data = block_reduce(in_data, factor, np.mean)
        else:  # upscale
            out_data = rescale(in_data, factor, order=1, preserve_range=True)

    try:
        out_data_array = Array(out_data,block.read_roi.offset,out_array.voxel_size)
        out_array[block.write_roi] = out_data_array.to_ndarray(block.write_roi)
    except Exception:
        print("Failed to write to %s" % block.write_roi)
        raise

    return 0

def scale_array(in_array, out_array, factor, write_size, mode):
    print(f"{mode.capitalize()}scaling by factor {factor}")

    dims = in_array.roi.dims
    context = write_size/8 if mode == 'up' else daisy.Coordinate((0,)*dims)
    write_block_roi = daisy.Roi((0,)*dims, write_size)
    read_block_roi = write_block_roi.grow(context,context)

    print("Processing ROI %s with block read_roi: %s, write_roi: %s" % (out_array.roi, read_block_roi, write_block_roi))

    task = daisy.Task(
        f'{mode.capitalize()}ScaleTask',
        out_array.roi.grow(context,context),
        read_block_roi,
        write_block_roi,
        process_function=partial(scale_block, in_array, out_array, factor, mode),
        read_write_conflict=True,
        num_workers=20,
        max_retries=0,
        fit='shrink')

    ret = daisy.run_blockwise([task])

    if ret:
        print("Ran all blocks successfully!")
    else:
        print("Did not run all blocks successfully...")

@click.command()
@click.option('--in_file', '-f', type=click.Path(exists=True), required=True, help="The path to the input zarr container")
@click.option('--in_ds_name', '-ds', type=str, required=True, help="The name of the input dataset")
@click.option('--scales', '-s', multiple=True, type=(int, int, int), help="The scale factors for each dimension", prompt="Enter the scale factors for each dimension")
@click.option('--chunk_shape', '-c', type=(int, int, int), default=None, help="The size of a chunk in voxels", prompt="Enter the chunk shape")
@click.option('--mode', '-m', type=click.Choice(['up', 'down']), required=True, prompt="Specify whether to upscale or downscale")
def scale_pyramid(in_file, in_ds_name, scales, chunk_shape, mode):
    """Create a scale pyramid of an array in a zarr container."""
    ds = zarr.open(in_file)

    # make sure in_ds_name points to a dataset
    try:
        prev_array = open_ds(os.path.join(in_file, in_ds_name))
    except Exception:
        raise RuntimeError("%s does not seem to be a dataset" % in_ds_name)

    if chunk_shape is not None:
        chunk_shape = daisy.Coordinate(chunk_shape)
    else:
        chunk_shape = daisy.Coordinate(prev_array.chunk_shape)
        print(f"Reusing chunk shape of {chunk_shape} for new datasets")
    
    # get scales 
    print(f"{mode.capitalize()}scaling by a factor of {scales}")

    # get ds_name
    match = re.search(r'/s(\d+)$', in_ds_name)
    if match:
        start_scale = int(match.group(1))
        if mode == 'down':
            ds_name = in_ds_name
            in_ds_name = in_ds_name[:-3]
        else: 
            if start_scale - len(scales) < 0:
                ds_name = in_ds_name[:-3] + f'/s{len(scales) - start_scale}'
                print(f"Renaming {in_ds_name} to {ds_name}, {start_scale}")
                ds.store.rename(in_ds_name, ds_name)
                in_ds_name = in_ds_name[:3]
            else: 
                ds_name = in_ds_name
                in_ds_name = in_ds_name[:-3]
    else:
        ds_name = in_ds_name + '/s0' if mode == 'down' else in_ds_name + f'/s{len(scales)}'
        print(f"Renaming {in_ds_name} to {ds_name}")
        ds.store.rename(in_ds_name, in_ds_name + '__tmp')
        ds.create_group(in_ds_name)
        ds.store.rename(in_ds_name + '__tmp', ds_name)
        start_scale = int(ds_name[-1])

    scale_numbers = [start_scale + (1 if mode == 'down' else -1) * i for i in range(1,1+len(scales))]
    prev_array = open_ds(os.path.join(in_file, ds_name))

    if prev_array.channel_dims == 0:
        num_channels = 1
    elif prev_array.channel_dims == 1:
        num_channels = prev_array.shape[0]
    else:
        raise RuntimeError("more than one channel not yet implemented, sorry...")

    for scale_num, scale in zip(scale_numbers,scales):
        try:
            scale = daisy.Coordinate(scale)
        except Exception:
            scale = daisy.Coordinate((scale,)*chunk_shape.dims)

        if mode == 'up':
            next_voxel_size = prev_array.voxel_size / scale
        else:  # downscale
            next_voxel_size = prev_array.voxel_size * scale
        
        next_ds_name = f"{in_ds_name}/s{scale_num}"
        next_write_size = chunk_shape * next_voxel_size
        next_total_roi = prev_array.roi.snap_to_grid(next_voxel_size, mode='grow')

        print(f"Next voxel size: {next_voxel_size}")
        print(f"Next total ROI: {next_total_roi}")
        print(f"Next chunk size: {next_write_size}")
        print(f"Preparing {next_ds_name}")

        next_array = prepare_ds(
            os.path.join(in_file, next_ds_name),
            shape=next_total_roi.shape / next_voxel_size,
            offset=next_total_roi.offset,
            voxel_size=next_voxel_size,
            axis_names=prev_array.axis_names,
            units=prev_array.units,
            dtype=prev_array.dtype,
            chunk_shape=chunk_shape,
        )

        scale_array(prev_array, next_array, scale, next_write_size, mode)

        prev_array = next_array

    print("Scale pyramid creation completed.")

if __name__ == "__main__":
    scale_pyramid()