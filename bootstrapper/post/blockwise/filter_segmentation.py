from tqdm import tqdm
import time
import sys
import toml
from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
import daisy
import numpy as np
from functools import partial, reduce


def filter_in_block(
    in_labels, out_labels, out_mask, error_mask, ids_to_remove, erode_out_mask, block
):
    import numpy as np
    from funlib.persistence import Array
    from skimage.morphology import ball, disk
    from scipy.ndimage import binary_erosion

    # Read labels
    labels_array = in_labels.to_ndarray(block.read_roi, fill_value=0)
    # labels_array = in_labels[block.read_roi]

    # Apply filtering
    mask_out = np.isin(labels_array, ids_to_remove)
    labels_array[mask_out] = 0

    # Create object mask
    mask_array = labels_array > 0

    # Apply LSD error mask if provided
    if error_mask is not None:
        error_mask_array = error_mask.to_ndarray(block.read_roi)
        mask_array *= np.logical_not(error_mask_array > 0)

    # Erode out mask in z if specified
    if erode_out_mask:
        z_struct = np.stack([ball(1)[0], ball(1)[0], np.zeros_like(disk(1))])
        mask_array = binary_erosion(mask_array, z_struct)

    # Write
    labels_array = Array(labels_array, block.read_roi.offset, in_labels.voxel_size)
    mask_array = Array(mask_array, block.read_roi.offset, in_labels.voxel_size)

    # crop
    labels_array = Array(
        labels_array[block.write_roi], block.write_roi.offset, in_labels.voxel_size
    )
    mask_array = Array(
        mask_array[block.write_roi], block.write_roi.offset, in_labels.voxel_size
    )

    out_labels[block.write_roi] = labels_array[block.write_roi]
    out_mask[block.write_roi] = mask_array[block.write_roi].astype(np.uint8)


def compute_ids_to_remove(
    labels_array, dust_filter, remove_outliers, remove_z_fragments=1, overlap_filter=0.0
):

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
        unique_ids_by_slice = [
            np.unique(labels_array[z]) for z in range(labels_array.shape[0])
        ]
        # Find IDs that exist in at least N z-slices
        N = remove_z_fragments
        z_id_counts = np.array(
            [
                np.sum([uid in slice_ids for slice_ids in unique_ids_by_slice])
                for uid in tqdm(filtered_ids)
            ]
        )
        filtered_ids = filtered_ids[z_id_counts >= N]
        print(f"After z-fragment removal: {len(filtered_ids)} ids")

    if overlap_filter > 0.0:

        st = time.time()
        # Compute areas for each ID in each slice
        areas = np.array([(labels_array == id).sum(axis=(1, 2)) for id in filtered_ids])

        # Compute intersections between adjacent slices
        intersections = np.array(
            [
                ((labels_array[:-1] == id) & (labels_array[1:] == id)).sum(axis=(1, 2))
                for id in filtered_ids
            ]
        )

        # Compute overlap ratios
        overlap_ratios = np.divide(
            intersections,
            areas[:, 1:],
            out=np.zeros_like(intersections, dtype=float),
            where=areas[:, 1:] != 0,
        )

        # An ID is kept if it has sufficient overlap in all slice pairs where it appears
        filtered_mask = np.all(
            np.logical_or(overlap_ratios >= overlap_filter, areas[:, 1:] == 0), axis=1
        )
        filtered_ids = filtered_ids[filtered_mask]
        print(
            f"After overlap filter: {len(filtered_ids)} ids, took {time.time() - st}s"
        )

    to_remove = np.setdiff1d(all_ids, filtered_ids)
    print(f"Total IDs to remove: {len(to_remove)}")
    return list(to_remove)


def filter_segmentation(
    seg_dataset,
    out_labels_dataset,
    out_mask_dataset,
    error_mask_dataset=None,
    dust_filter=0,
    remove_outliers=False,
    remove_z_fragments=1,
    overlap_filter=0.0,
    exclude_ids=None,
    erode_out_mask=False,
    roi_offset=None,
    roi_shape=None,
    block_shape=None,
    context=None,
    num_workers=20,
):

    # Open input dataset
    in_labels = open_ds(seg_dataset)
    voxel_size = in_labels.voxel_size

    # get total ROI
    if roi_offset is not None:
        total_roi = Roi(roi_offset, roi_shape)
    else:
        total_roi = in_labels.roi

    # get block size, context
    if context is not None:
        context = Coordinate(context) * voxel_size
    else:
        context = (
            Coordinate(
                [
                    2,
                ]
                * in_labels.roi.dims
            )
            * voxel_size
        )

    if block_shape == "roi":
        block_size = total_roi.get_shape()
        context = Coordinate(
            [
                0,
            ]
            * in_labels.roi.dims
        )
        num_workers = 1
    elif block_shape is None:
        block_size = Coordinate(in_labels.chunk_shape) * voxel_size
    else:
        block_size = Coordinate(block_shape) * voxel_size

    if error_mask_dataset is not None:
        error_mask = open_ds(error_mask_dataset)
    else:
        error_mask = None

    # Prepare output datasets
    out_labels = prepare_ds(
        out_labels_dataset,
        shape=total_roi.shape / in_labels.voxel_size,
        offset=total_roi.offset,
        voxel_size=in_labels.voxel_size,
        axis_names=in_labels.axis_names,
        units=in_labels.units,
        dtype=in_labels.dtype,
        chunk_shape=block_size / in_labels.voxel_size,
        mode="w",
    )

    out_mask = prepare_ds(
        out_mask_dataset,
        shape=total_roi.shape / in_labels.voxel_size,
        offset=total_roi.offset,
        voxel_size=in_labels.voxel_size,
        axis_names=in_labels.axis_names,
        units=in_labels.units,
        dtype=np.uint8,
        chunk_shape=block_size / in_labels.voxel_size,
        mode="w",
    )

    print(
        f"Filtering input segmentation {seg_dataset} into {out_labels_dataset} and {out_mask_dataset}"
    )

    read_roi = Roi((0,) * in_labels.roi.dims, block_size).grow(context, context)
    write_roi = Roi((0,) * in_labels.roi.dims, block_size)

    # Pre-compute global statistics
    print("Computing global statistics...")
    start = time.time()
    to_remove = compute_ids_to_remove(
        in_labels[total_roi],
        dust_filter,
        remove_outliers,
        remove_z_fragments,
        overlap_filter,
    )
    print(f"Computed global stats in {time.time() - start} \n")

    if exclude_ids is not None:
        for exclude_id in exclude_ids:
            if exclude_id not in to_remove:
                to_remove.append(exclude_id)

    # Create a daisy task
    filter_task = daisy.Task(
        "FilterSegmentationTask",
        total_roi.grow(context, context),
        read_roi,
        write_roi,
        process_function=partial(
            filter_in_block,
            in_labels,
            out_labels,
            out_mask,
            error_mask,
            to_remove,
            erode_out_mask,
        ),
        fit="shrink",
        num_workers=num_workers,
        read_write_conflict=True,
    )

    # Run the task
    success = daisy.run_blockwise([filter_task])

    if not success:
        raise RuntimeError("Filtering segmentation failed for at least one block")


if __name__ == "__main__":
    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = toml.load(f)

    filter_segmentation(**config)
