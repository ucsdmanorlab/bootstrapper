import argparse
import sys
import os
import numpy as np
import daisy
from funlib.persistence import open_ds, prepare_ds
from functools import partial

def make_raw_mask(in_ds, out_ds, block):
    import numpy as np
    from funlib.persistence import Array
    from skimage.morphology import binary_closing, disk

    in_data = in_ds.to_ndarray(block.read_roi, fill_value=0)

    footprint = np.stack([
        np.zeros_like(disk(10)),
        disk(10),
        np.zeros_like(disk(10))], axis=0)

    out_data = binary_closing(in_data, footprint)
    out_data = binary_closing(out_data, footprint)
    out_data = (out_data).astype(np.uint8)

    try:
        out_array = Array(out_data,block.read_roi,out_ds.voxel_size)
        out_ds[block.write_roi] = out_array.to_ndarray(block.write_roi)
    except Exception:
        print("Failed to write to %s" % block.write_roi)
        raise

    return 0

def make_obj_mask(in_ds, out_ds, block):
    import numpy as np
    from funlib.persistence import Array

    in_data = in_ds.to_ndarray(block.read_roi, fill_value=0)

    out_data = in_data > 0
    out_data = (out_data).astype(np.uint8)

    try:
        out_array = Array(out_data,block.read_roi,out_ds.voxel_size)
        out_ds[block.write_roi] = out_array.to_ndarray(block.write_roi)
    except Exception:
        print("Failed to write to %s" % block.write_roi)
        raise

    return 0


def mask_array(f, in_ds_name, out_ds_name, mode):
    
    # open
    in_ds = open_ds(f, in_ds_name)
    
    # prepare
    dims = in_ds.roi.dims
    write_size = in_ds.chunk_shape * in_ds.voxel_size
    context = write_size/8 if mode == 'img' else daisy.Coordinate((0,)*dims)
    write_block_roi = daisy.Roi((0,)*dims, write_size)
    read_block_roi = write_block_roi.grow(context,context)

    out_ds = prepare_ds(
        f,
        out_ds_name,
        total_roi=in_ds.roi,
        voxel_size=in_ds.voxel_size,
        dtype=np.uint8,
        write_size=write_size,
        force_exact_write_size=True,
        compressor=dict(id='blosc'))
    
    print("Processing ROI %s with block read_roi: %s, write_roi: %s" % (out_ds.roi, read_block_roi, write_block_roi))

    # run
    task = daisy.Task(
        f'{mode.capitalize()}MaskTask',
        out_ds.roi.grow(context,context),
        read_block_roi,
        write_block_roi,
        process_function=partial(make_raw_mask if mode == 'img' else make_obj_mask, in_ds, out_ds),
        read_write_conflict=True,
        num_workers=20,
        max_retries=0,
        fit='shrink')

    ret = daisy.run_blockwise([task])

    if ret:
        print("Ran all blocks successfully!")
    else:
        print("Did not run all blocks successfully...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a mask of a zarr image or labels array dataset.")

    parser.add_argument(
        '--file', '-f', type=str, help="The input container")
    parser.add_argument(
        '--input', '-i', type=str, help="The name of the input dataset")
    parser.add_argument(
        '--output', '-o', type=str, help="The name of the output dataset")
    parser.add_argument(
        '--mode', '-m', type=str, choices=['img', 'obj'], required=True,
        help="Specify whether to generate image or object mask")

    args = parser.parse_args()

    mask_array(args.file, args.input, args.output, args.mode)
