import json
import logging
import os
import sys

import gunpowder as gp
import numpy as np
import pytorch_lightning as pl
import toml
import torch
from funlib.persistence import open_ds

from bootstrapper.gp import (
    SmoothAugment,
    CreateMask,
    Renumber,
    DefectAugment,
    GammaAugment,
    ImpulseNoiseAugment,
)
from bootstrapper.training import GunpowderDataset, SnapshotCallback, fit
from model import Model, WeightedMSELoss

logging.getLogger().setLevel(logging.INFO)
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def create_pipeline(voxel_size, net_config, samples):
    batch_size = 1

    # array keys
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    unlabelled = gp.ArrayKey("UNLABELLED")

    gt_affs = gp.ArrayKey("GT_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    gt_affs_mask = gp.ArrayKey("AFFS_MASK")

    # batch keys for training
    keys = {
        "raw": raw,
        "gt_affs": gt_affs,
        "affs_weights": affs_weights,
    }

    # get affs task params
    neighborhood = net_config["outputs"]["3d_affs"]["neighborhood"]
    aff_grow_boundary = net_config["outputs"]["3d_affs"]["grow_boundary"]

    shape_increase = [0, 0, 0]  # net_config["shape_increase"]
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]
    output_shape = [x + y for x, y in zip(shape_increase, net_config["output_shape"])]

    # prepare request
    voxel_size = gp.Coordinate(voxel_size)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)

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
        control_point_spacing=voxel_size * gp.Coordinate(voxel_size[-1], voxel_size[0], voxel_size[0]),
        jitter_sigma=voxel_size * 2,
        spatial_dims=3,
        subsample=4,
        scale_interval=(0.9, 1.1),
        p=0.5,
    )
    pipeline += gp.ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=3, p=0.5)
    pipeline += gp.NoiseAugment(raw, p=0.5)
    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1,
        slab=(1, -1, -1),
        p=0.5,
    )
    pipeline += GammaAugment(raw, slab=(1, -1, -1), p=0.5)
    pipeline += ImpulseNoiseAugment(raw, pixel_p=0.05, p=0.5)
    pipeline += SmoothAugment(raw, p=0.5)
    pipeline += DefectAugment(
        raw, prob_missing=0.1, prob_low_contrast=0.1, prob_deform=0.0
    )

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

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(batch_size)

    return pipeline, request, keys


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.loss_fn = WeightedMSELoss()

    def forward(self, raw):
        return self.model(raw)

    def training_step(self, batch, batch_idx):
        pred_affs = self(batch["raw"])
        loss = self.loss_fn(pred_affs, batch["gt_affs"], batch["affs_weights"])
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "pred_affs": pred_affs.detach()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.5e-4)


def train(
    setup_dir,
    voxel_size,
    max_iterations,
    samples,
    save_checkpoints_every,
    save_snapshots_every,
):
    # load net config
    with open(os.path.join(setup_dir, "net_config.json")) as f:
        net_config = json.load(f)

    # prepare dataset
    dataset = GunpowderDataset(create_pipeline, voxel_size, net_config, samples)
    snapshot_callback = SnapshotCallback(setup_dir, voxel_size, save_snapshots_every)

    # train
    fit(
        LitModel(),
        dataset,
        setup_dir,
        max_iterations,
        save_checkpoints_every,
        snapshot_callback,
        num_workers=8,
    )


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = toml.load(f)

    assert config["setup_dir"] in setup_dir, "model directories do not match"
    config["setup_dir"] = setup_dir

    train(**config)
