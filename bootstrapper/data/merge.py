import click
import numpy as np
import json
import daisy
from funlib.persistence import open_ds, prepare_ds
from functools import partial
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_merge_block(in_ds, out_ds, luts, block):
    import numpy as np
    from funlib.segment.arrays import replace_values

    # load
    fragments = in_ds.to_ndarray(block.read_roi, fill_value=0)
    lut = luts['merges'] # {id: [ids]}
    old_vals = np.unique(fragments)
    new_vals = np.array([
        next((key for key, ids in lut.items() if val in ids), val) 
        for val in old_vals
    ]).astype(np.uint64)

    # replace values, write to empty array
    relabelled = np.zeros_like(fragments)
    relabelled = replace_values(
        in_array=fragments, old_values=old_vals, new_values=new_vals, out_array=relabelled
    )

    # write to segmentation array
    out_ds[block.write_roi] = relabelled


@click.command()
@click.option(
    "--in_seg",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="The path of the input segmentation zarr array",
)
@click.option(
    "--out_seg",
    "-o",
    type=click.Path(),
    help="The path of the output segmentation zarr array",
)
@click.option(
    "--luts",
    "-l",
    type=click.Path(exists=True),
    required=True,
    help="Path to the LUTs file",
)
def merge(in_seg, out_seg, luts):
    """
    Perform merges specified in the LUTs file on the input segmentation.

    Args:
        in_seg (str): Path to the input segmentation zarr array.
        out_seg (str): Path to the output segmentation zarr array.
        luts (str): Path to the LUTs file.
    """
    logger.info(f"Starting quick merge for {in_seg} to {out_seg} using {luts}")

    # open
    in_ds = open_ds(in_seg)
    with open(luts, "r") as f:
        luts_data = json.load(f)

    # prepare
    dims = in_ds.roi.dims
    block_size = in_ds.chunk_shape * in_ds.voxel_size
    context = daisy.Coordinate((0,) * dims)
    write_block_roi = daisy.Roi((0,) * dims, block_size)
    read_block_roi = write_block_roi.grow(context, context)

    if out_seg is None:
        in_f, in_ds_name = in_seg.split(".zarr")
        out_ds_name = in_ds_name+"__merged"
        out_ds = f"{in_f}.zarr/{out_ds_name}.zarr"
    else:
        out_ds = out_seg

    print(f"Writing to {out_ds}")
    out_ds = prepare_ds(
        out_ds,
        shape=in_ds.roi.shape / in_ds.voxel_size,
        offset=in_ds.roi.offset,
        voxel_size=in_ds.voxel_size,
        axis_names=in_ds.axis_names,
        units=in_ds.units,
        dtype=np.uint64,
        chunk_shape=in_ds.chunk_shape,
    )

    # run
    task = daisy.Task(
        f"MergeTask",
        out_ds.roi.grow(context, context),
        read_block_roi,
        write_block_roi,
        process_function=partial(
            quick_merge_block, in_ds, out_ds, luts_data
        ),
        read_write_conflict=False,
        num_workers=20,
        max_retries=0,
        fit="shrink",
    )

    ret = daisy.run_blockwise([task])

    if ret:
        logger.info("Ran all blocks successfully!")
    else:
        logger.error("Did not run all blocks successfully...")

    return out_ds


if __name__ == "__main__":
    merge()
