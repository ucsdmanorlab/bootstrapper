import click
import os
import re
import daisy
from funlib.persistence import open_ds, prepare_ds
import zarr
from functools import partial
import logging

from bootstrapper.blockwise import run_blockwise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LABEL_WORDS = ("label", "lbl", "ids", "mask", "seg")


def is_label_array(name, dtype):
    """True when the array holds object ids, which must never be averaged.

    Averaging id 5 and id 9 gives id 7, an object that never existed. The name
    decides for narrow dtypes, so a uint8 mask is still safe.
    """
    import numpy as np

    lowered = os.path.basename(name).lower()
    return dtype in (np.uint32, np.uint64) or any(w in lowered for w in LABEL_WORDS)


def scale_block(in_array, out_array, factor, mode, labels, block):
    import numpy as np
    from funlib.persistence import Array
    from skimage.measure import block_reduce
    from skimage.transform import rescale

    dims = len(factor)
    in_data = in_array.to_ndarray(block.read_roi, fill_value=0)

    n_channels = len(in_data.shape) - dims
    if n_channels >= 1:
        factor = (1,) * n_channels + factor

    if labels:
        if mode == "down":
            slices = tuple(slice(k // 2, None, k) for k in factor)
            out_data = in_data[slices]
        else:  # upscale
            out_data = in_data
            for axis, f in enumerate(factor):
                out_data = np.repeat(out_data, f, axis=axis)
    else:
        if mode == "down":
            out_data = block_reduce(in_data, factor, np.mean)
        else:  # upscale
            out_data = rescale(in_data, factor, order=1, preserve_range=True)

    try:
        out_data_array = Array(out_data, block.read_roi.offset, out_array.voxel_size)
        out_array[block.write_roi] = out_data_array.to_ndarray(block.write_roi)
    except Exception:
        logger.error(f"Failed to write to {block.write_roi}")
        raise

    return 0


def scale_array(in_array, out_array, factor, write_size, mode, labels):
    logger.info(f"{mode.capitalize()}scaling by factor {factor}")

    dims = in_array.roi.dims
    context = write_size / 8 if mode == "up" else daisy.Coordinate((0,) * dims)
    write_block_roi = daisy.Roi((0,) * dims, write_size)
    read_block_roi = write_block_roi.grow(context, context)

    logger.info(
        f"Processing ROI {out_array.roi} with block read_roi: {read_block_roi}, write_roi: {write_block_roi}"
    )

    task = daisy.Task(
        f"{mode.capitalize()}ScaleTask",
        out_array.roi.grow(context, context),
        read_block_roi,
        write_block_roi,
        process_function=partial(scale_block, in_array, out_array, factor, mode, labels),
        read_write_conflict=True,
        num_workers=20,
        max_retries=0,
        fit="shrink",
    )

    run_blockwise([task])
    logger.info("Ran all blocks successfully!")


def parse_factor(value):
    """Comma or space separated integers, as a tuple."""
    return tuple(int(v) for v in value.replace(",", " ").split())


@click.command()
@click.option(
    "--in_array",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="The path of the input zarr array, which may already end in a scale level",
    prompt="Enter the path to the input array",
)
@click.option(
    "--scales",
    "-s",
    multiple=True,
    required=True,
    type=str,
    help="Spatial scale factors for one level, e.g. 2,2,2. Repeat per level",
)
@click.option(
    "--chunk_shape",
    "-c",
    type=str,
    default=None,
    help="Spatial chunk shape in voxels, e.g. 64,64,64. Defaults to the input's",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["up", "down"]),
    required=True,
    prompt="Specify whether to upscale or downscale",
)
def scale_pyramid(in_array, scales, chunk_shape, mode):
    """
    Create a scale pyramid of a zarr array, in place.

    The array becomes s0 of a group of its own name, and every further level
    is coarser: s0 is always the finest. Upscaling counts down to s0 instead.

    Args:
        in_array (str): Path to the input zarr array.
        scales (tuple): Spatial scale factors, one string per level.
        chunk_shape (str): Spatial chunk shape, or None to reuse the input's.
        mode (str): 'up' for upscaling, 'down' for downscaling.
    """
    in_array = os.path.normpath(in_array)
    # the parent group, so no ".zarr" is needed anywhere in the path
    parent_path, ds_name = os.path.split(in_array)
    parent = zarr.open(parent_path)

    prev_array = open_ds(in_array)
    dims = prev_array.roi.dims
    channel_shape = tuple(prev_array.shape[: prev_array.channel_dims])

    scales = [parse_factor(s) for s in scales]
    for scale in scales:
        if len(scale) != dims:
            raise click.ClickException(
                f"Scale factor {scale} has {len(scale)} values, but "
                f"{in_array} has {dims} spatial dimensions."
            )

    if chunk_shape is not None:
        chunk_shape = daisy.Coordinate(parse_factor(chunk_shape))
        if chunk_shape.dims != dims:
            raise click.ClickException(
                f"Chunk shape {tuple(chunk_shape)} has {chunk_shape.dims} values, "
                f"but {in_array} has {dims} spatial dimensions."
            )
    else:
        chunk_shape = daisy.Coordinate(prev_array.chunk_shape[prev_array.channel_dims :])
        logger.info(f"Reusing chunk shape of {tuple(chunk_shape)} for new datasets")

    # the name the user gave the data, which an input already at a level carries
    # on its group instead
    at_level = re.match(r"^s(\d+)$", ds_name)
    pyramid_name = os.path.basename(parent_path) if at_level else ds_name
    labels = is_label_array(pyramid_name, prev_array.dtype)
    logger.info(
        f"{mode.capitalize()}scaling {in_array} by {scales} "
        f"({'labels: sampling' if labels else 'image: averaging'})"
    )

    # find the level this array already is, or make it one
    if at_level:
        start_scale = int(at_level.group(1))
        base_path = parent_path
        if mode == "up" and start_scale - len(scales) < 0:
            # no room below start_scale, so the input becomes the new top
            start_scale = len(scales)
            new_name = f"s{start_scale}"
            _refuse_if_exists(parent, new_name, base_path)
            logger.info(f"Renaming {ds_name} to {new_name}")
            parent.store.rename(ds_name, new_name)
            ds_name = new_name
    else:
        start_scale = 0 if mode == "down" else len(scales)
        base_path = in_array
        logger.info(f"Renaming {ds_name} to {ds_name}/s{start_scale}")
        parent.store.rename(ds_name, ds_name + "__tmp")
        parent.create_group(ds_name)
        parent.store.rename(ds_name + "__tmp", f"{ds_name}/s{start_scale}")
        parent = zarr.open(base_path)
        ds_name = f"s{start_scale}"

    scale_numbers = [
        start_scale + (1 if mode == "down" else -1) * i
        for i in range(1, 1 + len(scales))
    ]
    prev_array = open_ds(os.path.join(base_path, ds_name))

    for scale_num, scale in zip(scale_numbers, scales):
        scale = daisy.Coordinate(scale)

        if mode == "up":
            next_voxel_size = prev_array.voxel_size / scale
        else:  # downscale
            next_voxel_size = prev_array.voxel_size * scale

        next_name = f"s{scale_num}"
        _refuse_if_exists(parent, next_name, base_path)
        next_write_size = chunk_shape * next_voxel_size
        next_total_roi = prev_array.roi.snap_to_grid(next_voxel_size, mode="grow")

        logger.info(f"Next voxel size: {next_voxel_size}")
        logger.info(f"Next total ROI: {next_total_roi}")
        logger.info(f"Next chunk size: {next_write_size}")
        logger.info(f"Preparing {next_name}")

        next_array = prepare_ds(
            os.path.join(base_path, next_name),
            shape=channel_shape + tuple(next_total_roi.shape / next_voxel_size),
            offset=next_total_roi.offset,
            voxel_size=next_voxel_size,
            axis_names=prev_array.axis_names,
            units=prev_array.units,
            dtype=prev_array.dtype,
            chunk_shape=channel_shape + tuple(chunk_shape),
        )

        scale_array(prev_array, next_array, scale, next_write_size, mode, labels)

        prev_array = next_array

    logger.info("Scale pyramid creation completed.")


def _refuse_if_exists(group, name, base_path):
    if name in group:
        raise click.ClickException(
            f"{os.path.join(base_path, name)} already exists. "
            "Remove it or pick another input."
        )


if __name__ == "__main__":
    scale_pyramid()
