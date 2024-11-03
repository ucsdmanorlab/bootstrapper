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


class PrintArray(gp.BatchFilter):

    def __init__(self, array_key):
        self.array_key = array_key

    def process(self, batch, request):
        print(
            f"{self.array_key}: {batch[self.array_key].data.dtype}, {batch[self.array_key].data.shape}, np min, max, mean, median: {np.min(batch[self.array_key].data)}, {np.max(batch[self.array_key].data)}, {np.mean(batch[self.array_key].data)}, {np.median(batch[self.array_key].data)}"
        )


def compute_errors(
    seg_dataset,
    pred_dataset,
    mask_dataset,
    out_map_dataset,
    out_mask_dataset,
    thresholds=(0.1, 1.0),
    roi_offset=None,
    roi_shape=None,
    return_arrays=False,
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
    if roi_offset is not None:
        roi_offset = Coordinate(roi_offset)
        roi_shape  = Coordinate(roi_shape)
        roi = Roi(roi_offset, roi_shape).intersect(roi)
        
    # io shapes
    output_shape = Coordinate((8, 256, 256))
    input_shape = Coordinate((12, 384, 384))
    voxel_size = pred_ds.voxel_size

    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = calc_max_padding(output_size, voxel_size, 80)
    # context = (input_size - output_size) / 2

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
    logging.info(f"Prearing output datasets {out_map_dataset} and {out_mask_dataset}")

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
        sigma = 80  # TODO: unhardcode this

        pipeline += AddLSDErrors(
            seg,
            seg_pred,
            pred,
            error_map,
            error_mask,
            thresholds=thresholds,
            labels_mask=mask,
            sigma=sigma,
            downsample=2,
            array_specs=(
                {
                    error_map: gp.ArraySpec(
                        interpolatable=False,
                        voxel_size=voxel_size,
                        roi=total_output_roi,
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
            [0, 0, 8],
        ]  # TODO: unhardcode this

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
                        interpolatable=False,
                        voxel_size=voxel_size,
                        roi=total_output_roi,
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
    else:
        raise ValueError(f"Unknown prediction type: {pred_dataset}")

    pipeline += gp.IntensityScaleShift(error_map, 255, 0)
    pipeline += gp.AsType(error_map, np.uint8)

    pipeline += gp.ZarrWrite(
        dataset_names={error_map: out_map_dataset.split(".zarr")[-1]},
        store=out_map_dataset.split(".zarr")[0] + ".zarr",
    )

    pipeline += gp.ZarrWrite(
        dataset_names={error_mask: out_mask_dataset.split(".zarr")[-1]},
        store=out_mask_dataset.split(".zarr")[0] + ".zarr",
    )

    pipeline += gp.Scan(chunk_request)

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