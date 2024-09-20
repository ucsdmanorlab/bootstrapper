import sys
import os
import daisy
from functools import partial
from funlib.persistence import open_ds, prepare_ds


def block_fn(in_array, out_array, block):
    from funlib.persistence import Array
    from skimage.exposure import equalize_adapthist
    import numpy as np

    in_data = in_array.to_ndarray(block.read_roi)
    out_data = equalize_adapthist(in_data) * 255

    try:
        out_data_array = Array(out_data, block.read_roi.offset, out_array.voxel_size)
        out_array[block.write_roi] = out_data_array.to_ndarray(block.write_roi).astype(np.uint8)

    except Exception:
        print("Failed to write to %s" % block.write_roi)
        raise

    return 0

def run_blockwise(in_array, out_array, block_size, context):

    dims = in_array.roi.dims
    write_block_roi = daisy.Roi((0,)*dims, block_size)
    read_block_roi = write_block_roi.grow(context,context)

    print("Processing ROI %s with block read_roi: %s, write_roi: %s" % (out_array.roi, read_block_roi, write_block_roi))

    task = daisy.Task(
        'TestTask',
        out_array.roi.grow(context,context),
        read_block_roi,
        write_block_roi,
        process_function=partial(block_fn, in_array, out_array),
        read_write_conflict=False,
        num_workers=40,
        max_retries=0,
        fit='shrink')

    ret = daisy.run_blockwise([task])

    if ret:
        print("Ran all blocks successfully!")
    else:
        print("Did not run all blocks successfully...")


if __name__ == "__main__":
    in_f = sys.argv[1]
    in_ds = sys.argv[2]
    out_f = sys.argv[3]
    out_ds = sys.argv[4]

    in_array = open_ds(os.path.join(in_f, in_ds))
    out_array = prepare_ds(
        os.path.join(out_f, out_ds), 
        in_array.roi.shape / in_array.voxel_size, 
        in_array.roi.offset,
        in_array.voxel_size, 
        in_array.axis_names,
        in_array.units,
        in_array.chunk_shape,
        in_array.dtype)
    
    block_size = (200, 800, 800)
    context = (0, 400, 400)
    
    run_blockwise(in_array, out_array, block_size, context)