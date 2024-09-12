from funlib.persistence import open_ds, prepare_ds
import numpy as np
import os
from skimage.measure import regionprops, label
import sys

def filter_labels(in_f, in_ds, out_f, out_ds, size_threshold):
    """
    Perform a simple size filter on the labels in the input dataset.
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

    outlier_labels = [
        label
        for label, size in label_size_dict.items()
        if size <= size_threshold
    ]

    new_labels_array[np.isin(new_labels_array, outlier_labels)] = 0

    new_labels[new_labels.roi] = label(new_labels_array, connectivity=1).astype(labels.dtype)


if __name__ == "__main__":
    in_f = sys.argv[1]
    in_ds = sys.argv[2]
    out_f = sys.argv[3]
    out_ds = sys.argv[4]

    try:
        size_threshold = int(sys.argv[5])
    except:
        size_threshold = 500

    filter_labels(in_f, in_ds, out_f, out_ds, size_threshold)
