from tqdm import tqdm
import time
import sys
import yaml
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
import daisy
import numpy as np
from functools import partial, reduce


def filter_in_block(in_labels, out_labels, out_mask, lsd_errors, ids_to_remove, erode_out_mask, block):
    import numpy as np
    from funlib.persistence import Array
    from skimage.morphology import ball, disk
    from scipy.ndimage import binary_erosion
    from skimage.measure import label

    # Read labels
    labels_array = in_labels.to_ndarray(block.read_roi, fill_value=0)

    # Apply filtering
    mask_out = np.isin(labels_array, ids_to_remove)
    labels_array[mask_out] = 0

    # Create object mask
    mask_array = labels_array > 0

    # Apply LSD error mask if provided
    if lsd_errors is not None:
        lsd_errors_mask = lsd_errors.to_ndarray(block.read_roi)
        mask_array *= np.logical_not(lsd_errors_mask > 0)

    # Erode out mask in z if specified
    if erode_out_mask:
        z_struct = np.stack([ball(1)[0], ball(1)[0], np.zeros_like(disk(1))])
        mask_array = binary_erosion(mask_array, z_struct)

    # Write
    labels_array = Array(labels_array, block.read_roi, in_labels.voxel_size)
    mask_array = Array(mask_array, block.read_roi, in_labels.voxel_size)
    
    out_labels[block.write_roi] = labels_array.to_ndarray(block.write_roi)
    out_mask[block.write_roi] = mask_array.to_ndarray(block.write_roi).astype(np.uint8)


def compute_ids_to_remove(labels, dust_filter, remove_outliers, remove_z_fragments=1, overlap_filter=0.0):

    print("reading")
    labels_array = labels.to_ndarray()
    st = time.time()
    all_ids, id_counts = np.unique(labels_array, return_counts=True)
    print(f"reading time: {time.time() - st}, total_ids: {len(all_ids)}")
    
    # Initialize filtered_ids with all non-zero IDs
    filtered_ids = all_ids[all_ids != 0]
    
    if dust_filter > 0:
        # Filter by size
        filtered_ids = filtered_ids[id_counts[all_ids != 0] >= dust_filter]
        print(f"After size filter: {len(filtered_ids)} ids")
    
    if remove_outliers:
        # Get mean and std of counts for surviving IDs
        surviving_counts = id_counts[np.isin(all_ids, filtered_ids)]
        mean, std = np.mean(surviving_counts), np.std(surviving_counts)
        filtered_ids = filtered_ids[np.abs(surviving_counts - mean) <= 4 * std]
        print(f"After outlier removal: {len(filtered_ids)} ids")
    
    if remove_z_fragments > 1:
        # Find unique IDs by z-slice
        unique_ids_by_slice = [np.unique(labels_array[z]) for z in range(labels_array.shape[0])]
        # Find IDs that exist in at least N z-slices
        N = remove_z_fragments
        z_id_counts = np.array([np.sum([uid in slice_ids for slice_ids in unique_ids_by_slice]) for uid in tqdm(filtered_ids)])
        filtered_ids = filtered_ids[z_id_counts >= N]
        print(f"After z-fragment removal: {len(filtered_ids)} ids")
    
    if overlap_filter > 0.0:

        st = time.time()
        # Compute areas for each ID in each slice
        areas = np.array([(labels_array == id).sum(axis=(1, 2)) for id in filtered_ids])
        
        # Compute intersections between adjacent slices
        intersections = np.array([((labels_array[:-1] == id) & (labels_array[1:] == id)).sum(axis=(1, 2)) 
                                  for id in filtered_ids])
        
        # Compute overlap ratios
        overlap_ratios = np.divide(intersections, areas[:, 1:], 
                                   out=np.zeros_like(intersections, dtype=float), 
                                   where=areas[:, 1:] != 0)
        
        # An ID is kept if it has sufficient overlap in all slice pairs where it appears
        filtered_mask = np.all(np.logical_or(overlap_ratios >= overlap_filter, areas[:, 1:] == 0), axis=1)
        filtered_ids = filtered_ids[filtered_mask]
        print(f"After overlap filter: {len(filtered_ids)} ids, took {time.time() - st}s")

    to_remove = np.setdiff1d(all_ids, filtered_ids)
    print(f"Total IDs to remove: {len(to_remove)}")
    return list(to_remove)


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
        remove_z_fragments=1,
        overlap_filter=0.0,
        exclude_ids=None,
        erode_out_mask=False,
        block_size=None,
        context=None,
        num_workers=20
):

    # Open input dataset
    in_labels = open_ds(seg_file, seg_dataset)

    # Define block size and ROIs
    if block_size is None:
        block_size = in_labels.chunk_shape * in_labels.voxel_size
    if context is None:
        context = in_labels.chunk_shape * in_labels.voxel_size / 4

    if lsd_error_file is not None:
        lsd_errors = open_ds(lsd_error_file, lsd_error_mask_dataset)
    else:
        lsd_errors = None

    # Prepare output datasets
    out_labels = prepare_ds(
        out_file,
        out_labels_dataset,
        in_labels.roi,
        in_labels.voxel_size,
        in_labels.dtype,
        write_size=block_size,
        compressor={"id": "blosc"}
    )
    out_mask = prepare_ds(
        out_file,
        out_mask_dataset,
        in_labels.roi,
        in_labels.voxel_size,
        np.uint8,
        write_size=block_size,
        compressor={"id": "blosc"}
    )

    read_roi = Roi((0,)*in_labels.roi.dims, block_size).grow(context, context)
    write_roi = Roi((0,)*in_labels.roi.dims, block_size)

    # Pre-compute global statistics
    print("Computing global statistics...")
    start = time.time()
    to_remove = compute_ids_to_remove(in_labels, dust_filter, remove_outliers, remove_z_fragments, overlap_filter)
    print(f"Computed global stats in {time.time() - start} \n")

    if exclude_ids is not None:
        for exclude_id in exclude_ids:
            if exclude_id not in to_remove:
                to_remove.append(exclude_id)

    # Create a daisy task
    filter_task = daisy.Task(
        'FilterSegmentationTask',
        in_labels.roi.grow(context,context),
        read_roi,
        write_roi,
        process_function=partial(filter_in_block, in_labels, out_labels, out_mask, lsd_errors, to_remove, erode_out_mask),
        fit='shrink',
        num_workers=num_workers
    )

    # Run the task
    success = daisy.run_blockwise([filter_task])

    if not success:
        raise RuntimeError("Filtering segmentation failed for at least one block")


if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    config = yaml_config["processing"]["filter"]
    filter_segmentation(**config)
