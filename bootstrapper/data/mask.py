import click
import numpy as np
import daisy
import zarr
from funlib.persistence import open_ds, prepare_ds
from functools import partial
from scipy import ndimage
import logging

from bootstrapper.blockwise import run_blockwise
from bootstrapper.data.paths import mask_dataset_path

logger = logging.getLogger(__name__)


def make_raw_mask(in_ds, out_ds, block):
    import numpy as np
    from funlib.persistence import Array
    from scipy.ndimage import binary_dilation, binary_erosion
    from skimage.morphology import disk

    in_data = in_ds.to_ndarray(block.read_roi, fill_value=0)

    footprint = np.stack(
        [np.zeros_like(disk(10)), disk(10), np.zeros_like(disk(10))], axis=0
    )

    out_data = in_data
    for _ in range(2):  # closing that keeps the block edge, as skimage's did
        out_data = binary_erosion(
            binary_dilation(out_data, footprint), footprint, border_value=1
        )
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


def training_mask(volume, out_array):
    """Write the mask a volume trains on, as its training_mask says. "labels": labels > 0,
    grown mask_dilation voxels (in xy with 2d_labels). "everywhere": ones over the labels,
    or ones on the labelled sections with 2d_labels. Any other value is a mask's path.
    Returns the mask path, or None when the trainer's own labels > 0 serves."""
    mode, dilation, xy = volume["training_mask"], volume.get("mask_dilation", 0), volume.get("2d_labels", False)
    if mode not in ("labels", "everywhere"):
        return mode
    if mode == "labels" and not dilation:
        return None
    print(f"Writing the training mask to {out_array}")
    labels = open_ds(volume["labels"])
    prepare_ds(out_array, shape=labels.shape, offset=labels.offset, voxel_size=labels.voxel_size,
               axis_names=labels.axis_names, units=labels.units, chunk_shape=labels.chunk_shape,
               dtype=np.uint8, mode="w", fill_value=int(mode == "everywhere" and not xy))
    src, dst = zarr.open(volume["labels"], "r"), zarr.open(out_array, "r+")
    if mode == "everywhere" and xy:
        for z in range(src.shape[0]):
            if src[z].any():
                dst[z] = 1
    elif mode == "labels":
        structure = ndimage.generate_binary_structure(3, 1)
        if xy:
            structure[[0, 2]] = False
        halo = 0 if xy else dilation
        step = max(dst.chunks[0], 2 * halo)
        for z0 in range(0, src.shape[0], step):
            lo, hi = max(z0 - halo, 0), min(z0 + step + halo, src.shape[0])
            grown = ndimage.binary_dilation(np.stack([src[z] > 0 for z in range(lo, hi)]), structure, dilation)
            dst[z0 : z0 + step] = grown[z0 - lo :][:step]
    return out_array


@click.command()
@click.option(
    "--in-array",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="The path of the input zarr array",
)
@click.option(
    "--out-array",
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
    context = daisy.Coordinate((0, 21, 21)) * in_ds.voxel_size if mode == "raw" else daisy.Coordinate((0,) * dims)
    write_block_roi = daisy.Roi((0,) * dims, block_size)
    read_block_roi = write_block_roi.grow(context, context)

    out_array = mask_dataset_path(in_array, out_array)

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

    run_blockwise([task])
    logger.info("Ran all blocks successfully!")

    return out_array


if __name__ == "__main__":
    mask()
