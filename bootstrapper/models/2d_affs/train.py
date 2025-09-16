import torch
import gunpowder as gp
from funlib.persistence import open_ds
from model import Model, WeightedMSELoss

import sys
import toml
import json
import logging
import numpy as np
import os

from bootstrapper.gp import SmoothAugment, CreateMask, Renumber, DefectAugment, GammaAugment, ImpulseNoiseAugment


logging.getLogger().setLevel(logging.INFO)
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

torch.backends.cudnn.benchmark = True


def train(
    setup_dir,
    voxel_size,
    max_iterations,
    samples,
    save_checkpoints_every,
    save_snapshots_every,
):
    # array keys
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    unlabelled = gp.ArrayKey("UNLABELLED")

    gt_affs = gp.ArrayKey("GT_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    gt_affs_mask = gp.ArrayKey("AFFS_MASK")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    # model training setup
    model = Model(stack_infer=True)
    model.train()
    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    batch_size = 10

    # load net config
    with open(os.path.join(setup_dir, "net_config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "net_config.json")
        )
        net_config = json.load(f)

    # get affs task params
    neighborhood = net_config["outputs"]["2d_affs"]["neighborhood"]
    neighborhood = [
        [0, *x] for x in neighborhood
    ]  # add z-dimension since pipeline is 3D
    aff_grow_boundary = net_config["outputs"]["2d_affs"]["grow_boundary"]

    in_channels = net_config["in_channels"]
    shape_increase = [0, 0]  # net_config["shape_increase"]
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]
    output_shape = [x + y for x, y in zip(shape_increase, net_config["output_shape"])]

    # prepare request
    voxel_size = gp.Coordinate(voxel_size)
    input_size = gp.Coordinate((in_channels, *input_shape)) * voxel_size
    output_size = gp.Coordinate((1, *output_shape)) * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)

    # prepare pipeline
    source = tuple(
        (
            (
                gp.ArraySource(raw, open_ds(sample["raw"]), True),
                gp.ArraySource(labels, open_ds(sample["labels"]), False),
                gp.ArraySource(unlabelled, open_ds(sample["mask"]), False),
            )
            + gp.MergeProvider()
            if "mask" in sample and sample["mask"] is not None
            else (
                gp.ArraySource(raw, open_ds(sample["raw"]), True),
                gp.ArraySource(labels, open_ds(sample["labels"]), False),
            )
            + gp.MergeProvider()
            + CreateMask(labels, unlabelled)
        )
        + gp.Normalize(raw)
        + Renumber(labels)
        + gp.AsType(labels, "uint32")
        + gp.Pad(raw, None)
        + gp.Pad(labels, None)
        + gp.RandomLocation()
        + gp.Reject(mask=unlabelled, min_masked=0.05)
        for sample in samples
    )

    pipeline = source + gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    pipeline += gp.DeformAugment(
        control_point_spacing=gp.Coordinate((voxel_size[-2] * 10, voxel_size[-1] * 10)),
        jitter_sigma=(2.0 * voxel_size[-2], 2.0 * voxel_size[-1]),
        spatial_dims=2,
        subsample=1,
        scale_interval=(0.9, 1.1),
        p=0.5,
    )

    pipeline += gp.NoiseAugment(raw, p=0.5)

    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1,
        z_section_wise=True,
        p=0.5,
    )

    pipeline += GammaAugment(raw, slab=(1, -1, -1))
    pipeline += ImpulseNoiseAugment(raw, p=0.1)

    pipeline += SmoothAugment(raw, p=0.5)

    pipeline += DefectAugment(raw, prob_missing=0.0 if in_channels==1 else 0.05, prob_low_contrast=0.1)

    pipeline += gp.GrowBoundary(labels, mask=unlabelled, steps=aff_grow_boundary, only_xy=True)

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        unlabelled=unlabelled,
        affinities_mask=gt_affs_mask,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights, mask=gt_affs_mask)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache()

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={0: raw},
        loss_inputs={
            0: pred_affs,
            1: gt_affs,
            2: affs_weights,
        },
        outputs={
            0: pred_affs,
        },
        log_dir=os.path.join(setup_dir, "log"),
        checkpoint_basename=os.path.join(setup_dir, "model"),
        save_every=save_checkpoints_every,
    )

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)

    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
            affs_weights: "affs_weights",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=os.path.join(setup_dir, "snapshots"),
        every=save_snapshots_every,
    )

    # pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(max_iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = toml.load(f)

    assert config["setup_dir"] in setup_dir, "model directories do not match"
    config["setup_dir"] = setup_dir

    train(**config)
