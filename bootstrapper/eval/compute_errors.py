import time
import json
import logging
import sys
import numpy as np
import yaml
import os
import pprint

import gunpowder as gp
from funlib.persistence import prepare_ds, open_ds
from funlib.geometry import Coordinate, Roi

from bootstrapper.gp import AddLSDErrors, AddAffErrors, calc_max_padding

logging.basicConfig(level=logging.INFO)


def compute_errors(
    seg_dataset,
    pred_dataset,
    mask_dataset,
    out_map_dataset,
    out_mask_dataset,
    thresholds=(0.1, 1.0),
    return_arrays=False,
    num_workers=1,
):

    # array keys
    seg = gp.ArrayKey("SEGMENTATION")
    pred = gp.ArrayKey("MODEL_PRED")
    seg_pred = gp.ArrayKey("SEG_PRED")
    error_map = gp.ArrayKey("PRED_ERROR_MAP")
    error_mask = gp.ArrayKey("PRED_ERROR_MASK")

    pred_ds = open_ds(pred_dataset)
    seg_ds = open_ds(seg_dataset)

    if mask_dataset is not None:
        mask = gp.ArrayKey("MASK")
        mask_ds = open_ds(mask_dataset)
        mask_roi = mask_ds.roi
    else:
        mask = None
        mask_roi = pred_ds.roi

    # get rois
    roi = pred_ds.roi.intersect(seg_ds.roi).intersect(mask_roi)
    print(
        f"seg roi: {seg_ds.roi}, pred_roi: {pred_ds.roi}, mask_roi: {mask_roi}, intersection: {roi}"
    )

    # io shapes
    output_shape = Coordinate(pred_ds.chunk_shape[1:])
    input_shape = Coordinate(pred_ds.chunk_shape[1:]) * 2
    voxel_size = pred_ds.voxel_size

    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = calc_max_padding(output_size, voxel_size, 80)
    #context = (input_size - output_size) / 2

    total_input_roi = roi.grow(context, context)
    total_output_roi = roi

    # request
    chunk_request = gp.BatchRequest()
    chunk_request.add(seg, input_size)
    chunk_request.add(pred, input_size)
    if mask is not None:
        chunk_request.add(mask, input_size)
    chunk_request.add(seg_pred, input_size)
    chunk_request.add(error_map, output_size)
    chunk_request.add(error_mask, output_size)

    # output datasets
    prepare_ds(
        out_map_dataset,
        shape=total_output_roi.shape / voxel_size,
        offset=total_output_roi.offset,
        voxel_size=voxel_size,
        axis_names=seg_ds.axis_names,
        units=seg_ds.units,
        dtype=np.uint8,
    )

    prepare_ds(
        out_mask_dataset,
        shape=total_output_roi.shape / voxel_size,
        offset=total_output_roi.offset,
        voxel_size=voxel_size,
        axis_names=seg_ds.axis_names,
        units=seg_ds.units,
        dtype=np.uint8,
    )

    # pipeline
    sources = (gp.ArraySource(seg, seg_ds, False), gp.ArraySource(pred, pred_ds, True))
    if mask is not None:
        sources += (gp.ArraySource(mask, mask_ds, False),)

    pipeline = sources + gp.MergeProvider()

    pipeline += gp.Pad(seg, context)
    pipeline += gp.Pad(pred, None)
    if mask is not None:
        pipeline += gp.Pad(mask, None)
    pipeline += gp.Normalize(pred)

    if "3d_lsds" in pred_dataset and "_from_" not in pred_dataset:
        sigma = 80  #TODO: unhardcode this

        pipeline += AddLSDErrors(
            seg,
            seg_pred,
            pred,
            error_map,
            error_mask,
            thresholds=thresholds,
            labels_mask=mask,
            sigma=sigma,
            downsample=4,
            array_specs=(
                {
                    error_map: gp.ArraySpec(
                        interpolatable=False, voxel_size=voxel_size, roi=total_output_roi
                    ),
                    error_mask: gp.ArraySpec(
                        interpolatable=False,
                        voxel_size=voxel_size,
                        roi=total_output_roi,
                        dtype=np.uint8,
                    ),
                }
                if not return_arrays
                else None
            ),
        )
    elif "3d_affs" in pred_dataset:
        neighborhood = [
            [1, 0, 0], 
            [0, 1, 0], 
            [0, 0, 1], 
            [2, 0, 0], 
            [0, 8, 0], 
            [0, 0, 8]
        ] #TODO: unhardcode this

        pipeline += AddAffErrors(
            seg,
            seg_pred,
            pred,
            error_map,
            error_mask,
            neighborhood,
            thresholds=thresholds,
            labels_mask=mask,
            array_specs=(
                {
                    error_map: gp.ArraySpec(
                        interpolatable=False, voxel_size=voxel_size, roi=total_output_roi
                    ),
                    error_mask: gp.ArraySpec(
                        interpolatable=False,
                        voxel_size=voxel_size,
                        roi=total_output_roi,
                        dtype=np.uint8,
                    ),
                }
                if not return_arrays
                else None
            )
        )

    else:
        raise ValueError(f"Unknown prediction type: {pred_dataset}")

    pipeline += gp.IntensityScaleShift(error_map, 255, 0)
    pipeline += gp.AsType(error_map, np.uint8)

    pipeline += gp.ZarrWrite(
        dataset_names={error_map: out_map_dataset},
        store=out_map_dataset.split(".zarr")[0] + ".zarr",
    )

    pipeline += gp.ZarrWrite(
        dataset_names={error_mask: out_mask_dataset},
        store=out_mask_dataset.split(".zarr")[0] + ".zarr",
    )

    pipeline += gp.Scan(chunk_request, num_workers=num_workers)

    # request
    predict_request = gp.BatchRequest()

    if return_arrays:
        predict_request[error_map] = total_output_roi
        predict_request[error_mask] = total_output_roi

        with gp.build(pipeline):
            batch = pipeline.request_batch(predict_request)

        return batch

    else:
        with gp.build(pipeline):
            pipeline.request_batch(predict_request)


def compute_stats(array):

    total_voxels = int(np.prod(array.shape))
    num_nonzero_voxels = array[array > 0].size
    mean = np.mean(array)
    std = np.std(array)

    return {
        "mean": mean,
        "std": std,
        "num_nonzero_voxels": num_nonzero_voxels,
        "total_voxels": total_voxels,
        "nonzero_ratio": num_nonzero_voxels / total_voxels,
    }


if __name__ == "__main__":

    config_file = sys.argv[1]
    try:
        scores_out_file = sys.argv[2]
    except IndexError:
        scores_out_file = None

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if config != {}:
        start = time.time()
        ret = compute_errors(**config)
        end = time.time()

        seconds = end - start
        logging.info(f"Total time to compute LSD errors: {seconds}")

        ret = {
            "LSD_ERROR_MAP": open_ds(
                os.path.join(config["out_file"], config["out_map_dataset"])
            ),
            "LSD_ERROR_MASK": open_ds(
                os.path.join(config["out_file"], config["out_mask_dataset"])
            ),
        }

        logging.info("Computing LSD Error statistics..")
        stats = {}

        # compute stats for each array
        for array_key, arr in ret.items():
            arr_data = arr.data
            scores = compute_stats(arr_data[:])
            stats[str(array_key)] = scores

        # save scores to file
        if scores_out_file is not None:
            with open(scores_out_file, "w") as f:
                json.dump(stats, f)

        pprint.pp(stats)
