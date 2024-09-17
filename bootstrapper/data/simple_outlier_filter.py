from funlib.persistence import open_ds, prepare_ds
import numpy as np
import os
from skimage.measure import regionprops, label
import sys

def filter_labels(in_f, in_ds, out_f, out_ds, sigma=3.0):
    """
    Perform a simple outlier filter on a label dataset.
    """

    labels = open_ds(os.path.join(in_f, in_ds))

    new_labels = prepare_ds(
        os.path.join(out_f, out_ds),
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

    new_labels_array[np.isin(new_labels_array, outlier_labels)] = 0

    new_labels[new_labels.roi] = label(new_labels_array, connectivity=1).astype(labels.dtype)


if __name__ == "__main__":
    in_f = sys.argv[1]
    in_ds = sys.argv[2]
    out_f = sys.argv[3]
    out_ds = sys.argv[4]

    try:
        sigma = float(sys.argv[5])
    except:
        sigma = 3.0

    filter_labels(in_f, in_ds, out_f, out_ds, sigma)
