import torch
import gunpowder as gp
from funlib.persistence import open_ds
from model import Model, WeightedMSELoss

import sys
import toml
import json
import logging
import os

from bootstrapper.gp import (
    SmoothAugment,
    Add2DLSDs,
    CreateMask,
    Renumber,
    calc_max_padding,
)

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

    gt_lsds = gp.ArrayKey("GT_LSDS")
    lsds_weights = gp.ArrayKey("LSDS_WEIGHTS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

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

    # get lsd task params
    sigma = net_config["outputs"]["2d_lsds"]["sigma"]
    sigma = (0, sigma, sigma)  # add z-dimension since pipeline is 3D
    lsd_downsample = net_config["outputs"]["2d_lsds"]["downsample"]

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
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(pred_lsds, output_size)

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
        control_point_spacing=(voxel_size[-1] * 10, voxel_size[-1] * 10),
        jitter_sigma=(3.0, 3.0),
        spatial_dims=2,
        subsample=1,
        scale_interval=(0.9, 1.1),
        graph_raster_voxel_size=voxel_size[1:],
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

    pipeline += SmoothAugment(raw, p=0.5)

    pipeline += gp.DefectAugment(raw, prob_missing=0.0 if in_channels==1 else 0.05)

    pipeline += Add2DLSDs(
        labels,
        gt_lsds,
        unlabelled=unlabelled,
        lsds_mask=lsds_weights,
        sigma=sigma,
        downsample=lsd_downsample,
    )

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache()

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={0: raw},
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: lsds_weights,
        },
        outputs={
            0: pred_lsds,
        },
        log_dir=os.path.join(setup_dir, "log"),
        checkpoint_basename=os.path.join(setup_dir, "model"),
        save_every=save_checkpoints_every,
    )

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)

    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
            gt_lsds: "gt_lsds",
            pred_lsds: "pred_lsds",
            lsds_weights: "lsds_weights",
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
