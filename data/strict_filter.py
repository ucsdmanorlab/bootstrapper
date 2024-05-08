import sys
from tqdm import tqdm
from funlib.persistence import open_ds, prepare_ds

import numpy as np
from skimage.measure import label

if __name__ == "__main__":
    # inputs
    in_seg_f, in_seg_ds, in_mask_f, in_mask_ds, out_f, out_seg_ds, out_mask_ds = sys.argv[1:]

    # open arrays
    labels = open_ds(in_seg_f, in_seg_ds)
    mask = open_ds(in_mask_f, in_mask_ds)

    # prepare output arrays
    new_labels = prepare_ds(
        out_f,
        out_seg_ds,
        labels.roi,
        labels.voxel_size,
        labels.dtype,
        compressor={"id": "blosc"},
    )
    new_mask = prepare_ds(
        out_f,
        out_mask_ds,
        mask.roi,
        mask.voxel_size,
        mask.dtype,
        compressor={"id": "blosc"},
    )

    # read
    print("reading")
    new_labels_array = labels.to_ndarray()
    new_mask_array = mask.to_ndarray()

    # Find unique IDs by z-slice
    print("getting unique ids by section")
    unique_ids_by_slice = [np.unique(new_labels_array[z]) for z in range(new_labels_array.shape[0])]

    # Find IDs that exist in atleast N z-slices
    N = 20
    print("finding z fragments")
    all_ids, id_counts = np.unique(new_labels_array, return_counts=True)
    z_id_counts = np.array([np.sum([uid in slice_ids for slice_ids in unique_ids_by_slice]) for uid in tqdm(all_ids)])
    z_fragments = all_ids[z_id_counts < N]

    # get sizes and filter by size
    #print("filtering labels by count")
    #mean_size = np.mean(id_counts)
    #to_remove = np.union1d(all_ids[id_counts < (0.5*mean_size)], z_fragments)
    to_remove = z_fragments

    #print(f"mean size: {mean_size}, removing {len(to_remove)} ids")
    print(f"removing {len(to_remove)} ids")

    # mask out
    mask_out = np.isin(new_labels_array, to_remove)
    new_mask_array[mask_out] = 0
    new_labels_array[mask_out] = 0

    # write
    print("writing")
    new_labels[labels.roi] = label(new_labels_array, connectivity=1).astype(np.uint64)
    new_mask[mask.roi] = new_mask_array
