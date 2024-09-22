import click
import daisy
import logging
from functools import partial
from funlib.persistence import open_ds, prepare_ds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def block_fn(in_array, out_array, block):
    from funlib.persistence import Array
    from skimage.exposure import equalize_adapthist
    import numpy as np

    in_data = in_array.to_ndarray(block.read_roi)
    out_data = equalize_adapthist(in_data) * 255

    try:
        out_data_array = Array(out_data, block.read_roi.offset, out_array.voxel_size)
        out_array[block.write_roi] = out_data_array.to_ndarray(block.write_roi).astype(
            np.uint8
        )
    except Exception as e:
        logger.error(f"Failed to write to {block.write_roi}: {str(e)}")
        raise

    return 0

def run_blockwise(in_array, out_array, block_shape, context):
    dims = in_array.roi.dims
    vs = in_array.voxel_size

    if dims != 3:
        logger.error("Only 3D data is supported.")
        raise ValueError("Only 3D data is supported.")

    block_size = daisy.Coordinate(block_shape) * vs
    context = daisy.Coordinate(context) * vs
    write_block_roi = daisy.Roi((0,) * dims, block_size)
    read_block_roi = write_block_roi.grow(context, context)

    logger.info(
        f"Processing ROI {out_array.roi} with block read_roi: {read_block_roi}, write_roi: {write_block_roi}"
    )

    task = daisy.Task(
        "ClaheTask",
        out_array.roi.grow(context, context),
        read_block_roi,
        write_block_roi,
        process_function=partial(block_fn, in_array, out_array),
        read_write_conflict=False,
        num_workers=10,
        max_retries=0,
        fit="shrink",
    )

    ret = daisy.run_blockwise([task])

    if ret:
        logger.info("Ran all blocks successfully!")
    else:
        logger.error("Did not run all blocks successfully...")

@click.command()
@click.option(
    "--in_arr",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to input Zarr array.",
)
@click.option("--out_arr", "-o", type=click.Path(), help="Path to output Zarr array.")
@click.option(
    "--block-shape",
    "-b",
    type=(int, int, int),
    default=(1, 256, 256),
    help="Block shape in voxels as a tuple of integers.",
    show_default=True,
)
@click.option(
    "--context",
    "-c",
    type=(int, int, int),
    default=(0, 128, 128),
    help="Context in voxels as a tuple of integers.",
    show_default=True,
)
def clahe(in_arr, out_arr, block_shape, context):
    """
    Run Contrast Limited Adaptive Histogram Equalization (CLAHE) on a Zarr array.

    Args:
        in_arr (str): Path to input Zarr array.
        out_arr (str): Path to output Zarr array.
        block_shape (tuple): Block shape in voxels as a tuple of integers.
        context (tuple): Context in voxels as a tuple of integers.

    Returns:
        str: Path to the output Zarr array.
    """
    logger.info(f"Starting CLAHE processing on {in_arr}")
    in_array = open_ds(in_arr)

    # TODO: Add a check for the number of dimensions and handle block_shape and context

    if out_arr is None:
        out_arr = in_arr.replace(".zarr", "_clahe.zarr")

    out_array = prepare_ds(
        out_arr,
        in_array.roi.shape / in_array.voxel_size,
        in_array.roi.offset,
        in_array.voxel_size,
        in_array.axis_names,
        in_array.units,
        in_array.chunk_shape,
        in_array.dtype,
    )

    run_blockwise(in_array, out_array, block_shape, context)
    print(f"Output created at {out_arr}")

    return out_arr

if __name__ == "__main__":
    clahe()
