import os
import re

import click

SCALE_LEVEL = re.compile(r"^s\d+$")


def mask_dataset_path(in_array, out_array=None):
    """
    Resolve the output path of the mask of an input array.

    A mask is its own array, so a scale level masks into the same level of a
    mask array: raw/s0 gives raw_mask/s0, never raw/s0_mask.

    Args:
        in_array (str): Path to the input zarr array.
        out_array (str, optional): Path given by the user, derived from in_array if None.

    Returns:
        str: Path to the output mask array.

    Raises:
        click.ClickException: If the output path is the input array.
    """
    if out_array is None:
        group, name = os.path.split(os.path.normpath(in_array))
        if SCALE_LEVEL.match(name):
            out_array = os.path.join(group + "_mask", name)
        else:
            out_array = os.path.normpath(in_array) + "_mask"

    if os.path.realpath(out_array) == os.path.realpath(in_array):
        raise click.ClickException(
            "Refusing to write the mask over its input: "
            f"output path {out_array} is the same array as input path {in_array}."
        )

    return out_array
