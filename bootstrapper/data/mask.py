import click
import numpy as np
import daisy
from funlib.persistence import open_ds, prepare_ds
from functools import partial
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_raw_mask(in_ds, out_ds, block):
    import numpy as np
    from funlib.persistence import Array
    from skimage.morphology import binary_closing, disk

    logger.info(f"Processing raw mask for block: {block}")

    in_data = in_ds.to_ndarray(block.read_roi, fill_value=0)

    footprint = np.stack(
        [np.zeros_like(disk(10)), disk(10), np.zeros_like(disk(10))], axis=0
    )

    out_data = binary_closing(in_data, footprint)
    out_data = binary_closing(out_data, footprint)
    out_data = (out_data).astype(np.uint8)

    try:
        out_array = Array(out_data, block.read_roi.offset, out_ds.voxel_size)
        out_ds[block.write_roi] = out_array.to_ndarray(block.write_roi)
    except Exception as e:
        logger.error(f"Failed to write to {block.write_roi}: {str(e)}")
        raise

    return 0


def make_obj_mask(in_ds, out_ds, block):
    import numpy as np
    from funlib.persistence import Array

    logger.info(f"Processing object mask for block: {block}")

    in_data = in_ds.to_ndarray(block.read_roi, fill_value=0)

    out_data = in_data > 0
    out_data = (out_data).astype(np.uint8)

    try:
        out_array = Array(out_data, block.read_roi.offset, out_ds.voxel_size)
        out_ds[block.write_roi] = out_array.to_ndarray(block.write_roi)
    except Exception as e:
        logger.error(f"Failed to write to {block.write_roi}: {str(e)}")
        raise

    return 0


@click.command()
@click.option(
    "--in_array",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="The path of the input zarr array",
)
@click.option(
    "--out_array",
    "-o",
    type=click.Path(),
    help="The path of the output mask zarr array",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["raw", "labels"]),
    required=True,
    help="Specify whether to mask image or objects",
)
def mask(in_array, out_array, mode):
    """
    Generate a mask of a zarr image or labels array blockwise.

    Args:
        in_array (str): Path to the input zarr array.
        out_array (str): Path to the output mask zarr array.
        mode (str): Specify whether to mask image ("raw") or objects ("labels").

    Returns:
        str: Path to the output array.
    """
    logger.info(f"Starting mask generation for {mode} mode")

    # open
    in_ds = open_ds(in_array)

    # prepare
    dims = in_ds.roi.dims
    block_size = in_ds.chunk_shape * in_ds.voxel_size
    context = block_size / 8 if mode == "raw" else daisy.Coordinate((0,) * dims)
    write_block_roi = daisy.Roi((0,) * dims, block_size)
    read_block_roi = write_block_roi.grow(context, context)

    if out_array is None:
        in_f, in_ds_name = in_array.split(".zarr")
        out_ds = in_ds_name.replace(mode, f"{mode}_mask")
        out_array = f"{in_f}.zarr/{out_ds}"

    print(f"Writing mask to {out_array}")
    out_ds = prepare_ds(
        out_array,
        shape=in_ds.roi.shape / in_ds.voxel_size,
        offset=in_ds.roi.offset,
        voxel_size=in_ds.voxel_size,
        axis_names=in_ds.axis_names,
        units=in_ds.units,
        dtype=np.uint8,
        chunk_shape=in_ds.chunk_shape,
    )

    # run
    task = daisy.Task(
        f"{mode.capitalize()}MaskTask",
        out_ds.roi.grow(context, context),
        read_block_roi,
        write_block_roi,
        process_function=partial(
            make_raw_mask if mode == "raw" else make_obj_mask, in_ds, out_ds
        ),
        read_write_conflict=True,
        num_workers=20,
        max_retries=0,
        fit="shrink",
    )

    ret = daisy.run_blockwise([task])

    if ret:
        logger.info("Ran all blocks successfully!")
    else:
        logger.error("Did not run all blocks successfully...")

    return out_array


if __name__ == "__main__":
    make_mask()
