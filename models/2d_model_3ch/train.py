import torch
import gunpowder as gp
from model import MtlsdModel, WeightedMSELoss
from lsd.train.gp import AddLocalShapeDescriptor

import sys
import yaml
import json
import logging
import math
import numpy as np
import os

from utils import SmoothArray, CustomAffs, CustomLSDs


logging.basicConfig(level=logging.INFO)
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

torch.backends.cudnn.benchmark = True


def train(
        setup_dir,
        voxel_size,
        max_iterations,
        samples,
        raw_datasets,
        labels_datasets,
        mask_datasets,
        out_dir,
        save_checkpoints_every,
        save_snapshots_every,
):
    # array keys
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    unlabelled = gp.ArrayKey("UNLABELLED")
    
    gt_lsds = gp.ArrayKey("GT_LSDS")
    lsds_weights = gp.ArrayKey("LSDS_WEIGHTS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    
    gt_affs = gp.ArrayKey("GT_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    gt_affs_mask = gp.ArrayKey("AFFS_MASK")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    # model training setup 
    model = MtlsdModel(stack_infer=True)
    model.train()
    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4)
    batch_size = 10

    # load net config
    with open(os.path.join(setup_dir, "config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "config.json")
        )
        net_config = json.load(f)

    shape_increase = [0,0] #net_config["shape_increase"]
    input_shape = [x + y for x,y in zip(shape_increase,net_config["input_shape"])]
    output_shape = [x + y for x,y in zip(shape_increase,net_config["output_shape"])]

    # prepare samples
    samples = {
        samples[i]: {
            'raw': raw_datasets[i],
            'labels': labels_datasets[i],
            'mask': mask_datasets[i]
        }
        for i in range(len(samples))
    }

    # prepare request
    voxel_size = gp.Coordinate(voxel_size) 
    input_size = gp.Coordinate((3,*input_shape)) * voxel_size
    output_size = gp.Coordinate((1,*output_shape)) * voxel_size
    context = (input_size - output_size) / 2
    sigma = net_config['sigma']

    print(input_size, output_size, context)

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(pred_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)

    # pipeline
    source = tuple(
        gp.ZarrSource(
            sample,
            {
                raw: samples[sample]['raw'],
                labels: samples[sample]['labels'],
                unlabelled: samples[sample]['mask']
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
                labels: gp.ArraySpec(interpolatable=False),
                unlabelled: gp.ArraySpec(interpolatable=False),
            },
        )
        + gp.Normalize(raw)
        + gp.Pad(raw, None)
        + gp.Pad(labels, context)
        + gp.Pad(unlabelled, context)
        + gp.RandomLocation(mask=unlabelled, min_masked=0.05)
        for sample in samples
    )
    
    pipeline = source + gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1,2])

    pipeline += gp.DeformAugment(
        control_point_spacing=(voxel_size[0], voxel_size[0]),
        jitter_sigma=(2.0, 2.0),
        spatial_dims=2,
        subsample=1,
        scale_interval=(0.9, 1.1),
        graph_raster_voxel_size=voxel_size[1:],
    )
   
    pipeline += gp.DefectAugment(
            raw,
    )

    pipeline += gp.IntensityAugment(
        raw, scale_min=0.9, scale_max=1.1, shift_min=-0.1, shift_max=0.1, z_section_wise=True
    )

    #pipeline += SmoothArray(raw)

    pipeline += CustomLSDs(
            labels,
            gt_lsds,
            unlabelled=unlabelled,
            lsds_mask=lsds_weights,
            sigma=(0,sigma,sigma),
            downsample=2,
    )

    pipeline += gp.GrowBoundary(labels, mask=unlabelled, only_xy=True)

    pipeline += CustomAffs(
        affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        labels=labels,
        affinities=gt_affs,
        unlabelled=unlabelled,
        affinities_mask=gt_affs_mask,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights, mask=gt_affs_mask)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    #pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=40,cache_size=80)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"input": raw},
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: lsds_weights,
            3: pred_affs,
            4: gt_affs,
            5: affs_weights,
        },
        outputs={
            0: pred_lsds,
            1: pred_affs,
        },
        log_dir=os.path.join(out_dir,'log'),
        checkpoint_basename=os.path.join(out_dir,'model'),
        save_every=save_checkpoints_every,
    )

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)

    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
            gt_lsds: "gt_lsds",
            pred_lsds: "pred_lsds",
            lsds_weights: "lsds_weights",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
            affs_weights: "affs_weights",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=os.path.join(out_dir,'snapshots'),
        every=save_snapshots_every,
    )

    #pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(max_iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":

    config_file = sys.argv[1]
    model_name = sys.argv[2]
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["train"][model_name]

    assert config["setup_dir"] in setup_dir, \
        "model directories do not match"
    config["setup_dir"] = setup_dir

    assert len(config["samples"]) == len(config["raw_datasets"]) == \
        len(config["labels_datasets"]) == len(config["mask_datasets"]), \
        "number of samples and datasets do not match"

    train(**config)
