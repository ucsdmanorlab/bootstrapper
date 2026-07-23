import json
import logging
import os
import sys

import gunpowder as gp
import pytorch_lightning as pl
import toml
import torch
from funlib.persistence import open_ds

from bootstrapper.gp import (
    SmoothAugment,
    Add2DLSDs,
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
    batch_size = 10

    # array keys
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    unlabelled = gp.ArrayKey("UNLABELLED")

    gt_lsds = gp.ArrayKey("GT_LSDS")
    lsds_weights = gp.ArrayKey("LSDS_WEIGHTS")

    # batch keys for training
    keys = {
        "raw": raw,
        "gt_lsds": gt_lsds,
        "lsds_weights": lsds_weights,
    }

    # get lsd task params
    sigma = net_config["outputs"]["2d_lsds"]["sigma"]
    sigma = (0, sigma, sigma)  # add z-dimension since pipeline is 3D
    lsd_downsample = net_config["outputs"]["2d_lsds"]["downsample"]

    adj_slices = net_config["adj_slices"]
    shape_increase = [0, 0]  # net_config["shape_increase"]
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]
    output_shape = [x + y for x, y in zip(shape_increase, net_config["output_shape"])]

    # prepare request
    voxel_size = gp.Coordinate(voxel_size)
    input_size = gp.Coordinate((adj_slices, *input_shape)) * voxel_size
    output_size = gp.Coordinate((1, *output_shape)) * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)

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
        control_point_spacing=gp.Coordinate((voxel_size[-2] * voxel_size[0], voxel_size[-1] * voxel_size[0])),
        jitter_sigma=(2.0 * voxel_size[-2], 2.0 * voxel_size[-1]),
        spatial_dims=2,
        subsample=1,
        scale_interval=(0.9, 1.1),
        p=0.5,
    )
    if adj_slices > 1:
        pipeline += gp.ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=3)
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
    pipeline += DefectAugment(raw, prob_missing=0.0 if adj_slices==1 else 0.05, prob_low_contrast=0.1)

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

    return pipeline, request, keys


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model(stack_infer=True)
        self.loss_fn = WeightedMSELoss()

    def forward(self, raw):
        return self.model(raw)

    def training_step(self, batch, batch_idx):
        pred_lsds = self(batch["raw"])
        loss = self.loss_fn(pred_lsds, batch["gt_lsds"], batch["lsds_weights"])
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "pred_lsds": pred_lsds.detach()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1.0e-4)


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
