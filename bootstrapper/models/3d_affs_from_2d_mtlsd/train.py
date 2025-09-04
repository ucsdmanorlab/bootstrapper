import sys
import toml
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
    CustomGrowBoundary,
    ObfuscateAffs,
)

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

logging.getLogger().setLevel(logging.INFO)
torch.backends.cudnn.benchmark = True


def train(
    setup_dir=setup_dir,
    voxel_size=(1,1,1),
    max_iterations=3001,
    save_checkpoints_every=1000,
    save_snapshots_every=1000,
):
    batch_size = 10
    model = AffsUNet()
    model.train()
    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4)

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

    # get affs task params
    out_neighborhood = net_config["outputs"]["3d_affs"]["neighborhood"]
    out_aff_grow_boundary = net_config["outputs"]["3d_affs"]["grow_boundary"]
    in_neighborhood = net_config["inputs"]["2d_affs"]["neighborhood"]
    in_neighborhood = [
        [0, *x] for x in in_neighborhood
    ]  # add z-dimension since pipeline is 3D
    in_aff_grow_boundary = net_config["inputs"]["2d_affs"]["grow_boundary"]

    # get lsd task params
    sigma = net_config["inputs"]["2d_lsds"]["sigma"]
    sigma = (0, sigma, sigma)  # add z-dimension since pipeline is 3D
    lsd_downsample = net_config["inputs"]["2d_lsds"]["downsample"]

    shape_increase = [0, 0, 0]  # net_config["shape_increase"]
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]

    if "output_shape" not in net_config:
        output_shape = model.forward(
            input_affs=torch.empty(size=[1, 2] + input_shape),
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
    request.add(input_affs, input_size)
    request.add(input_lsds, input_size)
    request.add(gt_affs, output_size)
    request.add(pred_affs, output_size)
    request.add(affs_weights, output_size)

    # construct pipeline
    pipeline = CreateLabels(labels, shape=input_shape, voxel_size=voxel_size)

    pipeline += gp.Pad(labels, None, mode="reflect")

    pipeline += gp.DeformAugment(
        control_point_spacing=gp.Coordinate((voxel_size[-2] * 20, voxel_size[-1] * 20)),
        jitter_sigma=(3.0 * voxel_size[-2], 3.0 * voxel_size[-1]),
        spatial_dims=2,
        subsample=2,
        scale_interval=(0.9, 1.1),
        p=0.5,
    )

    pipeline += gp.ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=5)

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    pipeline += AddObfuscated2DLSDs(labels, input_lsds, sigma=sigma, downsample=lsd_downsample)

    pipeline += CustomGrowBoundary(labels, max_steps=in_aff_grow_boundary, only_xy=True)

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
    pipeline += gp.NoiseAugment(input_affs, mode="poisson", p=0.25)
    pipeline += gp.NoiseAugment(input_lsds, mode="gaussian", p=0.25)

    # intensity
    pipeline += gp.IntensityAugment(
        input_affs, 0.9, 1.1, -0.1, 0.1, z_section_wise=True, p=0.5
    )
    pipeline += gp.IntensityAugment(
        input_lsds, 0.9, 1.1, -0.1, 0.1, z_section_wise=True, p=0.5
    )

    # smooth the batch by different sigmas to simulate noisy predictions
    pipeline += SmoothAugment(input_affs, p=0.5)
    pipeline += SmoothAugment(input_lsds, p=0.5)

    # add defects
    pipeline += gp.DefectAugment(
        input_lsds, prob_missing=0.15, prob_low_contrast=0.05, prob_deform=0.0, axis=1
    )

    pipeline += gp.DefectAugment(
        input_affs, prob_missing=0.15, prob_low_contrast=0.05, prob_deform=0.0, axis=1
    )

    # now we erode - we want the gt affs to have a pixel boundary
    pipeline += gp.GrowBoundary(labels, steps=out_aff_grow_boundary, only_xy=True)

    pipeline += gp.AddAffinities(
        affinity_neighborhood=out_neighborhood,
        labels=labels,
        affinities=gt_affs,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights)

    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache()

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={
            0: input_lsds,
            1: input_affs,
        },
        loss_inputs={0: pred_affs, 1: gt_affs, 2: affs_weights},
        outputs={0: pred_affs},
        save_every=save_checkpoints_every,
        log_dir=os.path.join(setup_dir, "log"),
        checkpoint_basename=os.path.join(setup_dir, "model"),
    )

    # pipeline += gp.Squeeze([input_affs, input_lsds, gt_affs, pred_affs, affs_weights])

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
        output_dir=os.path.join(setup_dir, "snapshots"),
        every=save_snapshots_every,
    )

    with gp.build(pipeline):
        for i in range(max_iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":

    try:
        config_file = sys.argv[1]
        with open(config_file, "r") as f:
            config = toml.load(f)

        assert config["setup_dir"] in setup_dir, "model directories do not match"
        config["setup_dir"] = setup_dir

        train(**config)
    except:
        train()
