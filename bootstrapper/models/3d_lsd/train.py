import json
import logging
import os
import sys
import toml

import torch
import gunpowder as gp
from funlib.persistence import open_ds
from lsd.train.gp import AddLocalShapeDescriptor

from bootstrapper.gp import SmoothAugment, CreateMask, Renumber, calc_max_padding
from model import Model, WeightedMSELoss

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
    model = Model()
    model.train()
    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4)
    batch_size = 1

    # load net config
    with open(os.path.join(setup_dir, "net_config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "net_config.json")
        )
        net_config = json.load(f)

    # get lsd sigma
    sigma = net_config["outputs"]["3d_lsds"]["sigma"]

    shape_increase = [0, 0, 0]  # net_config["shape_increase"]
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]
    output_shape = [x + y for x, y in zip(shape_increase, net_config["output_shape"])]

    # prepare request
    voxel_size = gp.Coordinate(voxel_size)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size
    context = calc_max_padding(output_size, voxel_size, sigma)

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(unlabelled, output_size)
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
        + gp.Pad(labels, context)
        + gp.Pad(unlabelled, context)
        + gp.RandomLocation()
        + gp.Reject(mask=unlabelled, min_masked=0.05)
        for sample in samples
    )

    pipeline = source + gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    pipeline += gp.DeformAugment(
        control_point_spacing=(voxel_size[-1] * 10, voxel_size[-1] * 10),
        jitter_sigma=(2.0, 2.0),
        spatial_dims=2,
        subsample=1,
        scale_interval=(0.9, 1.1),
        graph_raster_voxel_size=voxel_size[1:],
        p=0.5,
    )

    pipeline += gp.ShiftAugment(prob_slip=0.1, prob_shift=0.1, sigma=1)

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

    pipeline += gp.DefectAugment(
        raw, prob_missing=0.05, prob_low_contrast=0.05, prob_deform=0.0
    )

    pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        unlabelled=unlabelled,
        lsds_mask=lsds_weights,
        sigma=sigma,
        downsample=2,
    )

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=80, cache_size=80)

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

    pipeline += gp.Squeeze([raw, gt_lsds, pred_lsds, lsds_weights])

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
