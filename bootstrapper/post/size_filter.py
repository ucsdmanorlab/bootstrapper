import logging
from funlib.persistence import open_ds, prepare_ds
import numpy as np
import click
from skimage.measure import regionprops, label

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
@click.option(
    "--in_array", "-i", required=True, type=str, help="Input labels zarr array"
)
@click.option("--out_array", "-o", type=str, help="Output labels zarr array")
@click.option(
    "--size_threshold",
    "-s",
    type=int,
    default=500,
    help="Size threshold in voxels for filtering",
    show_default=True,
)
def size_filter(in_array, out_array, size_threshold):
    """
    Perform a simple size filter on the labels in the input dataset.

    Args:
        in_array (str): Path to the input labels zarr array.
        out_array (str): Path to the output labels zarr array.
        size_threshold (int): Size threshold in voxels for filtering.

    Returns:
        None

    This function reads the input labels, filters out regions smaller than
    the specified threshold numher of voxels, and writes the result to the output array.
    """

    labels = open_ds(in_array)

    if out_array is None:
        out_array = in_array
        logging.info(f"Filtering in place at {out_array}")

    new_labels = prepare_ds(
        out_array,
        shape=labels.shape,
        offset=labels.offset,
        voxel_size=labels.voxel_size,
        axis_names=labels.axis_names,
        units=labels.units,
        dtype=labels.dtype,
        chunk_shape=labels.chunk_shape,
    )

    new_labels_array = labels.to_ndarray(labels.roi)

    regions = regionprops(new_labels_array)
    label_size_dict = {r.label: r.area for r in regions}

    outlier_labels = [
        label for label, size in label_size_dict.items() if size <= size_threshold
    ]

    logger.info(f"Filtered {len(outlier_labels)} IDs out of {len(label_size_dict)} IDs")

    new_labels_array[np.isin(new_labels_array, outlier_labels)] = 0

    new_labels[new_labels.roi] = label(new_labels_array, connectivity=1).astype(
        labels.dtype
    )

    print(f"Filtered output at {out_array}")
    return out_array


if __name__ == "__main__":
    size_filter()
