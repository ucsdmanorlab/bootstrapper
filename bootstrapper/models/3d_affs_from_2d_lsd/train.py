import sys
import yaml
import os
import json
import logging
import numpy as np
import torch
import gunpowder as gp

from model import AffsUNet, WeightedMSELoss
from bootstrapper.gp import (
    CreateLabels,
    AddObfuscated2DLSDs,
    SmoothAugment,
    CustomIntensityAugment,
)

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

logging.getLogger().setLevel(logging.INFO)
torch.backends.cudnn.benchmark = True


def train(
    setup_dir,
    voxel_size,
    max_iterations,
    save_checkpoints_every,
    save_snapshots_every,
):
    batch_size = 1
    model = AffsUNet()
    model.train()
    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4)

    labels = gp.ArrayKey("SYNTHETIC_LABELS")
    input_lsds = gp.ArrayKey("INPUT_2D_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

    with open(os.path.join(setup_dir, "net_config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "net_config.json")
        )
        net_config = json.load(f)

    # get affs neighborhoods
    out_neighborhood = net_config["outputs"]["3d_affs"]["neighborhood"]

    # get lsd sigma
    sigma = net_config["outputs"]["2d_lsds"]["sigma"]
    sigma = (0, sigma, sigma) # add z-dimension since pipeline is 3D

    shape_increase = [0, 0, 0]  # net_config["shape_increase"]
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]

    if "output_shape" not in net_config:
        output_shape = model.forward(
            input_lsds=torch.empty(size=[1, 6] + input_shape),
        )[0].shape[1:]
        net_config["output_shape"] = list(output_shape)
        with open(os.path.join(setup_dir, "net_config.json"), "w") as f:
            json.dump(net_config, f, indent=4)
    else:
        output_shape = [
            x + y for x, y in zip(shape_increase, net_config["output_shape"])
        ]
    print(output_shape)

    voxel_size = gp.Coordinate(voxel_size)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    request = gp.BatchRequest()
    request.add(labels, input_size)
    request.add(input_lsds, input_size)
    request.add(gt_affs, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)

    # construct pipeline
    pipeline = CreateLabels(labels, shape=input_shape, voxel_size=voxel_size)

    pipeline += gp.Pad(labels, None, mode="reflect")

    pipeline += gp.DeformAugment(
        control_point_spacing=(voxel_size[0], voxel_size[0]),
        jitter_sigma=(5.0, 5.0),
        spatial_dims=2,
        subsample=1,
        scale_interval=(0.9, 1.1),
        graph_raster_voxel_size=voxel_size[1:],
        p=0.5,
    )

    pipeline += gp.ShiftAugment(prob_slip=0.1, prob_shift=0.1, sigma=1)

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    # that is what predicted lsds will look like
    pipeline += AddObfuscated2DLSDs(labels, input_lsds, sigma=sigma, downsample=2)

    # add random noise
    pipeline += gp.NoiseAugment(input_lsds, mode="gaussian", p=0.5)

    # intensity
    pipeline += CustomIntensityAugment(
        input_lsds, 0.9, 1.1, -0.1, 0.1, z_section_wise=True, p=0.5
    )

    # smooth the batch by different sigmas to simulate noisy predictions
    pipeline += SmoothAugment(input_lsds, p=0.5)

    # add defects
    pipeline += gp.DefectAugment(
        input_lsds, prob_missing=0.05, prob_low_contrast=0.05, prob_deform=0.0, axis=1
    )

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
            0: input_lsds,
        },
        loss_inputs={0: pred_affs, 1: gt_affs, 2: affs_weights},
        outputs={0: pred_affs},
        save_every=save_checkpoints_every,
        log_dir=os.path.join(setup_dir, "log"),
        checkpoint_basename=os.path.join(setup_dir, "model"),
    )

    pipeline += gp.Squeeze([input_lsds, gt_affs, pred_affs, affs_weights])

    pipeline += gp.Snapshot(
        dataset_names={
            labels: "labels",
            input_lsds: "input_lsds",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
            affs_weights: "affs_weights",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=os.path.join(setup_dir, "snapshots"),
        every=save_snapshots_every,
    )

    with gp.build(pipeline):
        for i in range(max_iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    assert config["setup_dir"] in setup_dir, "model directories do not match"
    config["setup_dir"] = setup_dir

    train(**config)
