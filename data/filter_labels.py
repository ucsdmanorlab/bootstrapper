from funlib.persistence import open_ds, prepare_ds
import numpy as np
from skimage.measure import regionprops, label
import sys

if __name__ == "__main__":

    in_f = sys.argv[1]
    in_ds = sys.argv[2]

    out_ds = f"{in_ds}_filtered_relabeled"

    labels = open_ds(in_f,in_ds)

    new_labels = prepare_ds(
            in_f,
            out_ds,
            labels.roi,
            labels.voxel_size,
            labels.dtype,
            compressor={"id":"blosc"})

    new_labels_array = labels.to_ndarray()
    
    regions = regionprops(new_labels_array)
    label_size_dict = {r.label: r.area for r in regions}

    outlier_labels = [
        label
        for label, size in label_size_dict.items()
        if size <= 500
    ]

    new_labels_array[np.isin(new_labels_array, outlier_labels)] = 0

    new_labels[labels.roi] = label(new_labels_array,connectivity=1).astype(np.uint64)
