import time
import json
import logging
import sys
import numpy as np
import yaml
import os

from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import ball, disk

import gunpowder as gp
from funlib.persistence import prepare_ds, open_ds
from funlib.geometry import Coordinate, Roi
from lsd.train.gp import AddLocalShapeDescriptor

from lsd_error_stats import compute_stats 


logging.basicConfig(level=logging.INFO)


def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):

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


class ComputeDistance(gp.BatchFilter):

    def __init__(self, a, b, diff, mask):

        self.a = a
        self.b = b
        self.diff = diff
        self.mask = mask

    def setup(self):

        spec = self.spec[self.a].copy()
        self.provides(
            self.diff,
            spec)

    def process(self, batch, request):

        crop_roi = request[self.diff].roi

        a_data = batch[self.a].crop(crop_roi).data
        b_data = batch[self.b].crop(crop_roi).data
        mask_data = batch[self.mask].crop(crop_roi).data

        diff_data = np.sum((a_data - b_data)**2, axis=0)
        diff_data *= mask_data

        spec = batch[self.a].spec.copy()
        spec.roi = crop_roi

        # normalize
        epsilon = 1e-10  # a small constant to avoid division by zero
        max_value = np.max(diff_data)

        if max_value > epsilon:
            diff_data /= max_value
        else:
            diff_data[:] = 0

        batch[self.diff] = gp.Array(
            diff_data,
            spec
        )


class Threshold(gp.BatchFilter):

    def __init__(self, i, o, floor=0.0, ceil=1.0, grow=(0,0,0)):
        self.i = i # input array key
        self.o = o # output array key
        self.floor = floor
        self.ceil = ceil
        self.grow = Coordinate(grow)

    def setup(self):
        up_spec = self.spec[self.i].copy()
        up_spec.dtype = np.uint8
        self.provides(self.o,up_spec)

    def prepare(self, request):

        roi = request[self.i].roi

        grow = self.grow 
        grown_roi = roi.grow(grow,grow)
        grown_roi = request[self.i].roi.union(grown_roi)

        deps = gp.BatchRequest()
        deps[self.i] = request[self.i].copy()
        deps[self.i].roi = grown_roi
        return deps
    
    def process(self, batch, request):
        i_data = batch[self.i].data
        
        # threshold
        o_data = (i_data > self.floor) & (i_data < self.ceil)

        # dilate/erode
        z_struct = np.stack([ball(1)[0],]*3)
        xy_struct = np.stack([np.zeros((3,3)),disk(1),np.zeros((3,3))])

        # to join gaps between z-splits in error mask
        o_data = binary_dilation(o_data, z_struct)
        o_data = binary_erosion(o_data, z_struct)
        
        # to remove minor pixel-wise differences along xy boundaries
        o_data = binary_erosion(o_data, xy_struct, iterations=4)
        o_data = binary_dilation(o_data, xy_struct, iterations=4)

        o_data = o_data.astype(np.uint8)

        spec = batch[self.i].spec.copy()
        spec.dtype = np.uint8

        batch[self.o] = gp.Array(o_data,spec).crop(request[self.o].roi)


def compute_lsd_errors(
        seg_file,
        seg_dataset,
        lsds_file,
        lsds_dataset,
        mask_file,
        mask_dataset,
        out_file,
        out_map_dataset,
        out_mask_dataset,
        return_arrays=False):

    # array keys
    seg = gp.ArrayKey('SEGMENTATION')
    pred = gp.ArrayKey('PRED_LSDS')
    mask = gp.ArrayKey('MASK')
    seg_lsds = gp.ArrayKey('SEG_LSDS')
    error_map = gp.ArrayKey('LSD_ERROR_MAP')
    error_mask = gp.ArrayKey('ERROR_MASK')

    # get rois
    pred_ds = open_ds(lsds_file,lsds_dataset)
    pred_roi = pred_ds.roi
    seg_roi = open_ds(seg_file,seg_dataset).roi
    mask_roi = open_ds(mask_file,mask_dataset).roi
    roi = pred_roi.intersect(seg_roi).intersect(mask_roi)
    print(f"seg roi: {seg_roi}, pred_roi: {pred_roi}, mask_roi: {mask_roi}, intersection: {roi}")

    # io shapes
    xy_increase = 0
    input_shape = [32, 160 + xy_increase, 160 + xy_increase]
    output_shape = [8, 100 + xy_increase, 100 + xy_increase]

    voxel_size = pred_ds.voxel_size
    sigma = 80
    erode_grow = Coordinate((1,4,4)) * voxel_size
 
    voxel_size = Coordinate(voxel_size)
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = calc_max_padding(output_size, voxel_size, sigma)

    total_input_roi = roi.grow(context, context) 
    total_output_roi = roi

    # request 
    chunk_request = gp.BatchRequest()
    chunk_request.add(seg, input_size)
    chunk_request.add(pred, input_size)
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
        np.float32,
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
            ),
        gp.ZarrSource(
                mask_file,
                {mask: mask_dataset},
                {mask: gp.ArraySpec(interpolatable=False)}
            ),
    )

    pipeline = sources + gp.MergeProvider()

    pipeline += gp.Pad(seg, context)
    pipeline += gp.Normalize(pred)

    pipeline += AddLocalShapeDescriptor(
        seg,
        seg_lsds,
        sigma=sigma,
        downsample=4)
    
    pipeline += ComputeDistance(
        seg_lsds,
        pred,
        error_map,
        mask,
    )

    pipeline += Threshold(error_map, error_mask, floor=0.3, grow=erode_grow) # 90% confidence

    pipeline += gp.ZarrWrite(
            dataset_names={
                error_map: out_map_dataset,
                error_mask: out_mask_dataset
            },
            store=out_file
        )
    pipeline += gp.Scan(chunk_request, num_workers=20)

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
        

if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["processing"]["compute_lsd_errors"]

    start = time.time()
    arrays = compute_lsd_errors(**config).arrays
    end = time.time()

    seconds = end - start
    logging.info(f'Total time to compute LSD errors: {seconds}')

    arrays = {
            'lsd_error_map': open_ds(config['out_file'],config['out_map_dataset']),
            'lsd_error_mask': open_ds(config['out_file'],config['out_mask_dataset']),
    }

    if arrays is not None:
        logging.info("Computing LSD Error statistics..")
        stats = {}

        for array_key in arrays:
            arr = arrays[array_key].data
            stats[str(array_key)] = compute_stats(arr) 

        print(stats)
