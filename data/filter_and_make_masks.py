from funlib.persistence import open_ds, prepare_ds
import numpy as np
from skimage.measure import regionprops, label
import sys

if __name__ == "__main__":

    in_seg_f = sys.argv[1]
    in_seg_ds = sys.argv[2]
    in_mask_f = sys.argv[3]
    in_mask_ds = sys.argv[4]
    out_f = sys.argv[5]
    out_seg_ds = sys.argv[6]
    out_mask_ds = sys.argv[7]

    labels = open_ds(in_seg_f,in_seg_ds)
    mask = open_ds(in_mask_f,in_mask_ds)

    new_labels = prepare_ds(
            out_f,
            out_seg_ds,
            labels.roi,
            labels.voxel_size,
            labels.dtype,
            compressor={"id":"blosc"})
    
    new_mask = prepare_ds(
            out_f,
            out_mask_ds,
            mask.roi,
            mask.voxel_size,
            mask.dtype,
            compressor={"id":"blosc"})

    new_labels_array = labels.to_ndarray()
    new_mask_array = mask.to_ndarray()
    
    regions = regionprops(new_labels_array)
    label_size_dict = {r.label: r.area for r in regions}

    mean = np.mean(list(label_size_dict.values()))
    std_dev = np.std(list(label_size_dict.values()))

    outlier_labels = [
        label
        for label, size in label_size_dict.items()
        if size < mean
    ]

    print(f"removing {len(outlier_labels)} ids")

    new_mask_array[np.isin(new_labels_array, outlier_labels)] = 0
    new_labels_array[np.isin(new_labels_array, outlier_labels)] = 0

    new_labels[labels.roi] = label(new_labels_array,connectivity=1).astype(np.uint64)
    new_mask[mask.roi] = new_mask_array
