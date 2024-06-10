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

from add_lsd_errors import AddLSDErrors

logging.basicConfig(level=logging.INFO)


def calc_max_padding(output_size, voxel_size, sigma, mode="grow"):

    method_padding = Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = Roi(
        (
            Coordinate([i / 2 for i in [output_size[0], diag, diag]])
            + method_padding
        ),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


def compute_errors(
        seg_file,
        seg_dataset,
        lsds_file,
        lsds_dataset,
        mask_file,
        mask_dataset,
        out_file,
        out_map_dataset,
        out_mask_dataset,
        thresholds=(0.1,1.0),
        return_arrays=False):

    # array keys
    seg = gp.ArrayKey('SEGMENTATION')
    pred = gp.ArrayKey('PRED_LSDS')
    seg_lsds = gp.ArrayKey('SEG_LSDS')
    error_map = gp.ArrayKey('LSD_ERROR_MAP')
    error_mask = gp.ArrayKey('LSD_ERROR_MASK')
    
    pred_ds = open_ds(lsds_file,lsds_dataset)
    pred_roi = pred_ds.roi
    seg_roi = open_ds(seg_file,seg_dataset).roi
    
    if mask_file is not None:
        mask = gp.ArrayKey('MASK')
        mask_roi = open_ds(mask_file,mask_dataset).roi
    else:
        mask = None
        mask_roi = pred_roi

    # get rois
    roi = pred_roi.intersect(seg_roi).intersect(mask_roi)
    print(f"seg roi: {seg_roi}, pred_roi: {pred_roi}, mask_roi: {mask_roi}, intersection: {roi}")

    # io shapes
    output_shape = Coordinate(pred_ds.chunk_shape[1:])
    input_shape = Coordinate(pred_ds.chunk_shape[1:]) * 2

    voxel_size = pred_ds.voxel_size
    sigma = 80
 
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = calc_max_padding(output_size, voxel_size, sigma)

    total_input_roi = roi.grow(context, context) 
    total_output_roi = roi

    # request 
    chunk_request = gp.BatchRequest()
    chunk_request.add(seg, input_size)
    chunk_request.add(pred, input_size)
    if mask is not None:
        chunk_request.add(mask, input_size)
    chunk_request.add(seg_lsds, input_size)
    chunk_request.add(error_map, output_size)
    chunk_request.add(error_mask, output_size)

    # output datasets
    prepare_ds(
        out_file,
        out_map_dataset,
        total_output_roi,
        voxel_size,
        np.uint8,
        write_size=output_size,
        delete=True) 
    
    prepare_ds(
        out_file,
        out_mask_dataset,
        total_output_roi,
        voxel_size,
        np.uint8,
        write_size=output_size,
        delete=True) 

    # pipeline
    sources = (
        gp.ZarrSource(
                seg_file,
                {seg: seg_dataset},
                {seg: gp.ArraySpec(interpolatable=False)}
            ),
        gp.ZarrSource(
                lsds_file,
                {pred: lsds_dataset},
                {pred: gp.ArraySpec(interpolatable=True)}
            ),)
    if mask is not None:
        sources += (gp.ZarrSource(
                    mask_file,
                    {mask: mask_dataset},
                    {mask: gp.ArraySpec(interpolatable=False)}),)

    pipeline = sources + gp.MergeProvider()

    pipeline += gp.Pad(seg, None)
    pipeline += gp.Pad(pred, None)
    if mask is not None:
        pipeline += gp.Pad(mask, None)
    pipeline += gp.Normalize(pred)

    pipeline += AddLSDErrors(
        seg,
        seg_lsds,
        pred,
        error_map,
        error_mask,
        thresholds=thresholds,
        labels_mask=mask,
        sigma=sigma,
        downsample=4,
        array_specs={
            error_map: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size, roi=total_output_roi),
            error_mask: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size, roi=total_output_roi, dtype=np.uint8)
        } if not return_arrays else None
    )
   
    pipeline += gp.IntensityScaleShift(error_map, 255, 0)
    pipeline += gp.AsType(error_map, np.uint8)

    pipeline += gp.ZarrWrite(
            dataset_names={
                error_map: out_map_dataset,
                error_mask: out_mask_dataset
            },
            store=out_file
        )
    pipeline += gp.Scan(chunk_request, num_workers=40)

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
        'mean': mean,
        'std': std,  
        'num_nonzero_voxels': num_nonzero_voxels,
        'total_voxels': total_voxels,
        'nonzero_ratio' : num_nonzero_voxels / total_voxels,
    }


if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["processing"]["compute_lsd_errors"]

    if config != {}:
        start = time.time()
        ret = compute_errors(**config)
        end = time.time()

        seconds = end - start
        logging.info(f'Total time to compute LSD errors: {seconds}')

        # compute LSD error statistics
        ret = {
            'LSD_ERROR_MAP': (config['out_file'],config['out_map_dataset']),
            'LSD_ERROR_MASK': (config['out_file'],config['out_mask_dataset'])
        }

        logging.info("Computing LSD Error statistics..")
        stats = {}

        for array_key, val in ret.items():
            arr = open_ds(val[0],val[1])
            arr = ret[array_key].data
            scores = compute_stats(arr[:]) 
            stats[str(array_key)] = scores 

            scores_out_file = os.path.join(val[0],val[1],'.scores')
            logging.info(f"Writing scores for {array_key} to {scores_out_file}..")

            with open(scores_out_file, 'r+') as f:
                json.dump(scores, f, indent=4)

        pprint.pp(stats)
