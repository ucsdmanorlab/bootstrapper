import sys
import yaml
import os
import json
import logging
import math
import numpy as np
import random
import torch
import zarr
import gunpowder as gp

from model import AffsUNet, WeightedMSELoss
from utils import CreateLabels, CustomLSDs, SmoothAugment, NoiseAugment, IntensityAugment, CustomGrowBoundary, ObfuscateAffs

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

def init_weights(m):
    if isinstance(m, (torch.nn.Conv3d,torch.nn.ConvTranspose3d)):
        torch.nn.init.kaiming_normal_(m.weight,nonlinearity='relu')


def train(
        setup_dir,
        voxel_size,
        sigma,
        max_iterations,
        out_dir,
        save_checkpoints_every,
        save_snapshots_every,
):
    batch_size = 1 
    model = AffsUNet()
    model.apply(init_weights)
    model.train()
    loss = WeightedMSELoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.5e-4)

    labels = gp.ArrayKey("SYNTHETIC_LABELS")
    input_affs = gp.ArrayKey("INPUT_2D_AFFS")
    input_lsds = gp.ArrayKey("INPUT_2D_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    
    with open(os.path.join(setup_dir, "net_config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "net_config.json")
        )
        net_config = json.load(f)

    out_neighborhood = net_config["out_neighborhood"]
    in_neighborhood = net_config["in_neighborhood"]
    in_neighborhood = [[0,*x] for x in in_neighborhood] # add z-dimension since pipeline is 3D
    
    shape_increase = [0,0,0] #net_config["shape_increase"]
    input_shape = [x + y for x,y in zip(shape_increase,net_config["input_shape"])]
    
    if 'output_shape' not in net_config:
        output_shape = model.forward(
                input_affs=torch.empty(size=[1,2]+input_shape),
                input_lsds=torch.empty(size=[1,6]+input_shape),
            )[0].shape[1:]
        net_config['output_shape'] = list(output_shape)
        with open(os.path.join(setup_dir,"net_config.json"),"w") as f:
            json.dump(net_config,f,indent=4)
    else: 
        output_shape = [x + y for x,y in zip(shape_increase,net_config["output_shape"])]
    print(output_shape)

    voxel_size = gp.Coordinate(voxel_size) 
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size
    
    request = gp.BatchRequest()
    request.add(labels, input_size)
    request.add(input_affs, input_size)
    request.add(input_lsds, input_size)
    request.add(gt_affs, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)

    # construct pipeline
    pipeline = CreateLabels(
            labels,
            shape=input_shape,
            voxel_size=voxel_size)
            
    pipeline += gp.Pad(labels,None,mode="reflect")

    pipeline += gp.DeformAugment(
        control_point_spacing=(voxel_size[0], voxel_size[0]),
        jitter_sigma=(5.0, 5.0),
        spatial_dims=2,
        subsample=1,
        scale_interval=(0.9, 1.1),
        graph_raster_voxel_size=voxel_size[1:],
    )

    pipeline += gp.ShiftAugment(
        prob_slip=0.1,
        prob_shift=0.1,
        sigma=1)

    pipeline += gp.SimpleAugment(transpose_only=[1,2])
    
    pipeline += CustomLSDs(
        labels, input_lsds, sigma=sigma, downsample=2
    )

    pipeline += CustomGrowBoundary(labels, max_steps=1, only_xy=True)

    # that is what predicted affs will look like
    pipeline += gp.AddAffinities(
        affinity_neighborhood=in_neighborhood,
        labels=labels,
        affinities=input_affs,
        dtype=np.float32,
    )

    # add missing boundaries
    pipeline += ObfuscateAffs(input_affs)
    
    # add random noise
    pipeline += NoiseAugment(input_affs, mode='poisson')
    pipeline += NoiseAugment(input_lsds, mode='gaussian')

    # intensity
    pipeline += IntensityAugment(input_affs, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)
    pipeline += IntensityAugment(input_lsds, 0.9, 1.1, -0.1, 0.1, z_section_wise=True)

    # smooth the batch by different sigmas to simulate noisy predictions
    pipeline += SmoothAugment(input_affs, (0.5,1.5))
    pipeline += SmoothAugment(input_lsds, (0.5,1.5))

    # add defects
    pipeline += gp.DefectAugment(
        input_lsds, 
        prob_missing=0.05,
        prob_low_contrast=0.05,
        prob_deform=0.0,
        axis=1)
    
    pipeline += gp.DefectAugment(
        input_affs, 
        prob_missing=0.05,
        prob_low_contrast=0.05,
        prob_deform=0.0,
        axis=1)
    
    # now we erode - we want the gt affs to have a pixel boundary
    pipeline += gp.GrowBoundary(labels, steps=1, only_xy=True)

    pipeline += gp.AddAffinities(
        affinity_neighborhood=out_neighborhood,
        labels=labels,
        affinities=gt_affs,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights)

    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=32, cache_size=64)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={
            "input_lsds": input_lsds,
            "input_affs": input_affs,
        },
        loss_inputs={0: pred_affs, 1: gt_affs, 2: affs_weights},
        outputs={0: pred_affs},
        save_every=save_checkpoints_every,
        log_dir=os.path.join(setup_dir,'log'),
        checkpoint_basename=os.path.join(setup_dir,'model'),
    )

    pipeline += gp.Squeeze([input_affs,input_lsds,gt_affs,pred_affs,affs_weights])
    
    pipeline += gp.Snapshot(
        dataset_names={
            labels: "labels",
            input_affs: "input_affs",
            input_lsds: "input_lsds",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
            affs_weights: "affs_weights",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=os.path.join(setup_dir,'snapshots'),
        every=save_snapshots_every,
    )

    with gp.build(pipeline):
        for i in range(max_iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["3d_affs_from_2d_mtlsd"]

    assert config["setup_dir"] in setup_dir, \
        "model directories do not match"
    config["setup_dir"] = setup_dir
    config["out_dir"] = setup_dir
    
    train(**config)
