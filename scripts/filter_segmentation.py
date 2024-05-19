import sys
import yaml
from tqdm import tqdm
from funlib.persistence import open_ds, prepare_ds

import numpy as np
from functools import reduce
from skimage.measure import label
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import ball, disk


def filter_segmentation(
        seg_file,
        seg_dataset,
        out_file,
        out_labels_dataset,
        out_mask_dataset,
        lsd_error_file=None,
        lsd_error_mask_dataset=None,
        dust_filter=0,
        remove_outliers=False,
        remove_z_fragments=False,
        erode_out_mask=False,
):

    # open arrays
    labels = open_ds(seg_file, seg_dataset)

    # prepare output arrays
    new_labels = prepare_ds(
        out_file,
        out_labels_dataset,
        labels.roi,
        labels.voxel_size,
        labels.dtype,
        compressor={"id": "blosc"},
    )
    new_mask = prepare_ds(
        out_file,
        out_mask_dataset,
        labels.roi,
        labels.voxel_size,
        np.uint8,
        compressor={"id": "blosc"},
    )

    # read
    print("reading")
    new_labels_array = labels.to_ndarray()
    all_ids, id_counts = np.unique(new_labels_array, return_counts=True)

    size_filtered = np.array([0,])
    z_fragments = np.array([0,])

    if dust_filter > 0:
        # get sizes and filter by size
        #mean_size = np.mean(id_counts)
        size_filtered = all_ids[id_counts < dust_filter]
        print(f"size filtered: {len(size_filtered)}")

    if remove_outliers:
        # get mean and std of counts
        if dust_filter > 0:
            filtered_id_counts = id_counts[id_counts > dust_filter]
            mean, std = np.mean(filtered_id_counts), np.std(filtered_id_counts)
        else:
            mean, std = np.mean(id_counts), np.std(id_counts)
        
        outliers = all_ids[(np.abs(id_counts - mean) > 6 * std)]
        print(f"mean: {mean}, std: {std}, outliers: {len(outliers)}")

    if remove_z_fragments:
        # Find unique IDs by z-slice
        unique_ids_by_slice = [np.unique(new_labels_array[z]) for z in range(new_labels_array.shape[0])]

        # Find IDs that exist in atleast N z-slices
        N = 8
        z_id_counts = np.array([np.sum([uid in slice_ids for slice_ids in unique_ids_by_slice]) for uid in tqdm(all_ids)])
        z_fragments = all_ids[z_id_counts < N]
        print(f"z fragments: {len(z_fragments)}")
    
    to_remove = reduce(np.union1d,(size_filtered, z_fragments, outliers))
    print(f"removing {len(to_remove) - 1} ids")

    # mask out
    mask_out = np.isin(new_labels_array, to_remove)
    new_labels_array[mask_out] = 0

    # make out object mask
    new_mask_array = new_labels_array > 0 
    
    # combine with lsd errors
    if lsd_error_file is not None:
        print("applying lsd error mask")
        lsd_error_mask = open_ds(lsd_error_file, lsd_error_mask_dataset)
        new_mask_array *= np.logical_not(lsd_error_mask.to_ndarray() > 0)
    
    if erode_out_mask:
        print("eroding out mask")
        z_struct = np.stack([ball(1)[0],ball(1)[0],np.zeros_like(disk(1))]) # erode only in z in -1 direction
        new_mask_array = binary_erosion(new_mask_array, z_struct)

    # write
    print("writing")
    new_labels[labels.roi] = label(new_labels_array, connectivity=1).astype(np.uint64)
    new_mask[labels.roi] = new_mask_array.astype(np.uint8)


if __name__ == "__main__":
    
    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["processing"]["filter"]

    filter_segmentation(**config)
