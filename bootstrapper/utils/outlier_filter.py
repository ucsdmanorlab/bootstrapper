import click
from funlib.persistence import open_ds, prepare_ds
import numpy as np
from skimage.measure import regionprops, label


@click.command()
@click.option('--in_labels', '-i', required=True, type=str, help='Input labels zarr array')
@click.option('--out_labels', '-o', type=str, help='Output labels zarr array')
@click.option('--sigma', '-s', type=float, default=3.0, help='Outlier threshold in standard deviations', show_default=True)
def outlier_filter(in_labels, out_labels, sigma):
    """
    Perform a simple outlier filter on a label dataset.
    """

    labels = open_ds(in_labels)

    new_labels = prepare_ds(
        out_labels,
        shape=labels.shape,
        offset=labels.offset,
        voxel_size=labels.voxel_size,
        axis_names=labels.axis_names,
        units=labels.units,
        dtype=labels.dtype,
        chunk_shape=labels.chunk_shape
    )

    new_labels_array = labels.to_ndarray(labels.roi)

    regions = regionprops(new_labels_array)
    label_size_dict = {r.label: r.area for r in regions}

    mean, std = np.mean(list(label_size_dict.values())), np.std(list(label_size_dict.values()))

    outlier_labels = [
        label
        for label, size in label_size_dict.items()
        if abs(size - mean) > sigma * std
    ]

    click.echo(f"Found {len(outlier_labels)} outliers out of {len(label_size_dict)} IDs")

    new_labels_array[np.isin(new_labels_array, outlier_labels)] = 0

    new_labels[new_labels.roi] = label(new_labels_array, connectivity=1).astype(labels.dtype)

if __name__ == "__main__":
    outlier_filter()