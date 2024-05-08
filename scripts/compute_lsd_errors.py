import time
import yaml
import os
import gunpowder as gp
from funlib.persistence import prepare_ds, open_ds
from funlib.geometry import Coordinate, Roi

from lsd.train.gp import AddLocalShapeDescriptor

from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import ball, disk

import json
import logging
import sys
import numpy as np

logging.basicConfig(level=logging.INFO)


class ComputeDistance(gp.BatchFilter):

    def __init__(self, a, b, diff, mask):

        self.a = a
        self.b = b
        self.diff = diff
        self.mask = mask

    def setup(self):

        self.provides(
            self.diff,
            self.spec[self.a].copy())

    def prepare(self, request):
        
        deps = gp.BatchRequest()

        deps[self.a] = request[self.a].copy()
        deps[self.b] = request[self.b].copy()
        deps[self.mask] = request[self.mask].copy()
        
        return deps

    def process(self, batch, request):

        a_data = batch[self.a].data
        b_data = batch[self.b].data
        mask_data = batch[self.mask].data

        diff_data = np.sum((a_data - b_data)**2, axis=0)
        diff_data *= mask_data

        # normalize
        epsilon = 1e-10  # a small constant to avoid division by zero
        max_value = np.max(diff_data)

        if max_value > epsilon:
            diff_data /= max_value
        else:
            diff_data[:] = 0

        batch[self.diff] = gp.Array(
            diff_data,
            batch[self.a].spec.copy())


class Threshold(gp.BatchFilter):

    def __init__(self, i, o, floor=0.0, ceil=1.0):
        self.i = i # input array key
        self.o = o # output array key
        self.floor = floor
        self.ceil = ceil

    def setup(self):
        up_spec = self.spec[self.i].copy()
        up_spec.dtype = np.uint8
        self.provides(self.o,up_spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.i] = request[self.i].copy()
        return deps
    
    def process(self, batch, request):
        i_data = batch[self.i].data
        
        # threshold
        o_data = (i_data > self.floor) & (i_data < self.ceil)

        # dilate/erode
        z_struct = np.stack([ball(1)[0],]*3)
        xy_struct = np.stack([np.zeros((3,3)),disk(1),np.zeros((3,3))])

        o_data = binary_dilation(o_data, z_struct)
        o_data = binary_erosion(o_data, z_struct)
        
        o_data = binary_dilation(o_data, xy_struct, iterations=2)
        o_data = binary_erosion(o_data, xy_struct, iterations=4)
        o_data = binary_dilation(o_data, xy_struct, iterations=2)

        o_data = o_data.astype(np.uint8)

        spec = batch[self.i].spec.copy()
        spec.dtype = np.uint8

        batch[self.o] = gp.Array(o_data,spec)


def compute_lsd_errors(
        seg_file,
        seg_dataset,
        lsds_file,
        lsds_dataset,
        mask_file,
        mask_dataset,
        out_file,
        out_map_dataset,
        out_mask_dataset):

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
    input_shape = [20, 212 + 420, 212 + 420]
    output_shape = [4, 120 + 420, 120 + 420]
    voxel_size = pred_ds.voxel_size
    sigma = 80
 
    voxel_size = Coordinate(voxel_size)
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = (input_size - output_size) / 2

    total_input_roi = roi.grow(context, context) 
    total_output_roi = roi#.grow(-context,-context)

    # request 
    chunk_request = gp.BatchRequest()
    chunk_request.add(seg, input_size)
    chunk_request.add(pred, output_size)
    chunk_request.add(mask, output_size)
    chunk_request.add(seg_lsds, output_size)
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
                {seg: gp.ArraySpec(voxel_size=voxel_size, interpolatable=False, roi=seg_roi)}
            ),
        gp.ZarrSource(
                lsds_file,
                {pred: lsds_dataset},
                {pred: gp.ArraySpec(voxel_size=voxel_size, interpolatable=True, roi=pred_roi)}
            ),
        gp.ZarrSource(
                mask_file,
                {mask: mask_dataset},
                {mask: gp.ArraySpec(voxel_size=voxel_size, interpolatable=False, roi=mask_roi)}
            ),
    )

    pipeline = sources + gp.MergeProvider()

    pipeline += gp.Pad(seg, size=context, mode="reflect")
    pipeline += gp.Pad(pred, size=context, mode="reflect")
    pipeline += gp.Pad(mask, size=context)
    pipeline += gp.Normalize(pred)

    pipeline += AddLocalShapeDescriptor(
        seg,
        seg_lsds,
        sigma=sigma,
        downsample=2)
    
    pipeline += ComputeDistance(
        seg_lsds,
        pred,
        error_map,
        mask)

    pipeline += Threshold(error_map, error_mask, floor=0.1) # 90% confidence

    pipeline += gp.ZarrWrite(
            dataset_names={
                error_map: out_map_dataset,
                error_mask: out_mask_dataset
            },
            store=out_file
        )
    pipeline += gp.Scan(chunk_request, num_workers=80)

    predict_request = gp.BatchRequest()

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(predict_request)
    print("Prediction finished")


if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["processing"]["compute_lsd_errors"]

    start = time.time()
    compute_lsd_errors(**config)
    end = time.time()

    seconds = end - start
    logging.info(f'Total time to compute LSD errors: {seconds}')
