import os
import gunpowder as gp
from funlib.persistence import prepare_ds, open_ds
from funlib.geometry import Coordinate, Roi

from lsd.train.gp import AddLocalShapeDescriptor

import json
import logging
import sys
import numpy as np

logging.basicConfig(level=logging.INFO)

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
config_path = os.path.join(setup_dir,"config.json")

with open(config_path,"r") as f:
    config = json.load(f)


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

    def __init__(self, i, o, threshold=0.05, floor=0.0):
        self.i = i # input array key
        self.o = o # output array key
        self.threshold = threshold
        self.floor = floor

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
        o_data = (i_data > self.floor) & (i_data < self.threshold)
        o_data = o_data.astype(np.uint8)

        spec = batch[self.i].spec.copy()
        spec.dtype = np.uint8

        batch[self.o] = gp.Array(o_data,spec)


def compute_error_map(
        seg_file,
        seg_dataset,
        pred_file,
        pred_dataset,
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
    pred_roi = open_ds(pred_file,pred_dataset).roi
    seg_roi = open_ds(seg_file,seg_dataset).roi
    mask_roi = open_ds(mask_file,mask_dataset).roi
    roi = pred_roi.intersect(seg_roi).intersect(mask_roi)
    print(f"seg roi: {seg_roi}, pred_roi: {pred_roi}, mask_roi: {mask_roi}, intersection: {roi}")

    # io shapes
    shape_increase = config["shape_increase"]
    input_shape = [x + y for x,y in zip(shape_increase,config["input_shape"])]
    output_shape = [x + y for x,y in zip(shape_increase,config["output_shape"])]
    voxel_size = config["voxel_size"]
    sigma = config["sigma"]
 
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
                pred_file,
                {pred: pred_dataset},
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

    pipeline += Threshold(error_map, error_mask, threshold=0.1) # 90% confidence

    pipeline += gp.ZarrWrite(
            dataset_names={
                error_map: out_map_dataset,
                error_mask: out_mask_dataset
            },
            store=out_file
        )
    pipeline += gp.Scan(chunk_request, num_workers=40)

    predict_request = gp.BatchRequest()

    print("Starting prediction...")
    with gp.build(pipeline):
        pipeline.request_batch(predict_request)
    print("Prediction finished")
