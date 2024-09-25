import click
import logging
from funlib.persistence import open_ds, prepare_ds
import numpy as np
from skimage.measure import regionprops, label


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--in_labels",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input labels zarr array",
)
@click.option("--out_labels", "-o", type=str, help="Output labels zarr array")
@click.option(
    "--sigma",
    "-s",
    type=float,
    default=3.0,
    help="Outlier threshold in standard deviations",
)
def outlier_filter(in_labels, out_labels, sigma):
    """
    Perform a simple outlier filter on a label dataset.

    This function filters out outlier labels based on their size compared to the mean size of all labels.

    Args:
        in_labels (str): Path to the input labels zarr array.
        out_labels (str): Path to the output labels zarr array.
        sigma (float): Outlier threshold in standard deviations.

    Returns:
        None

    Example:
        >>> outlier_filter("input_labels.zarr", "output_labels.zarr", 3.0)
    """
    logger.info(f"Starting outlier filter with sigma={sigma}")

    labels = open_ds(in_labels)
    logger.info(f"Opened input labels from {in_labels}")

    if out_labels is None:
        out_labels = in_labels
        logging.info(f"Filtering in place at {in_labels}")

    new_labels = prepare_ds(
        out_labels,
        shape=labels.shape,
        offset=labels.offset,
        voxel_size=labels.voxel_size,
        axis_names=labels.axis_names,
        units=labels.units,
        dtype=labels.dtype,
        chunk_shape=labels.chunk_shape,
    )
    logger.info(f"Prepared output dataset at {out_labels}")

    new_labels_array = labels.to_ndarray(labels.roi)
    logger.info("Converted labels to numpy array")

    regions = regionprops(new_labels_array)
    label_size_dict = {r.label: r.area for r in regions}
    logger.info(f"Computed region properties for {len(regions)} regions")

    mean, std = np.mean(list(label_size_dict.values())), np.std(
        list(label_size_dict.values())
    )
    logger.info(f"Calculated mean size: {mean:.2f}, standard deviation: {std:.2f}")

    outlier_labels = [
        label
        for label, size in label_size_dict.items()
        if abs(size - mean) > sigma * std
    ]

    logger.info(
        f"Found {len(outlier_labels)} outliers out of {len(label_size_dict)} IDs"
    )

    new_labels_array[np.isin(new_labels_array, outlier_labels)] = 0
    logger.info("Removed outlier labels from the array")

    new_labels[new_labels.roi] = label(new_labels_array, connectivity=1).astype(
        labels.dtype
    )
    print(f"Filtered output at {out_labels}")
    return out_labels


if __name__ == "__main__":
    outlier_filter()
