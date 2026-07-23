import json
import logging
import os
import sys

import gunpowder as gp
import numpy as np
import pytorch_lightning as pl
import toml
import torch

from bootstrapper.gp import (
    CreateLabels,
    ObfuscateLabels,
    SmoothAugment,
    CustomGrowBoundary,
    DefectAugment,
)
from bootstrapper.training import GunpowderDataset, SnapshotCallback, fit
from model import AffsUNet, WeightedMSELoss

logging.getLogger().setLevel(logging.INFO)
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def create_pipeline(voxel_size, net_config):
    batch_size = 1

    # array keys
    labels = gp.ArrayKey("SYNTHETIC_LABELS")
    obfuscated_labels = gp.ArrayKey("OBFUSCATED_LABELS")
    input_affs = gp.ArrayKey("INPUT_2D_AFFS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

    # batch keys for training
    keys = {
        "input_affs": input_affs,
        "gt_affs": gt_affs,
        "affs_weights": affs_weights,
    }

    # get affs task params
    in_neighborhood = net_config["inputs"]["2d_affs"]["neighborhood"]
    in_neighborhood = [
        [0, *x] for x in in_neighborhood
    ]  # add z-dimension since pipeline is 3D
    in_grow_boundary = net_config["inputs"]["2d_affs"]["grow_boundary"]
    out_neighborhood = net_config["outputs"]["3d_affs"]["neighborhood"]
    out_aff_grow_boundary = net_config["outputs"]["3d_affs"]["grow_boundary"]

    shape_increase = [0, 0, 0]  # net_config["shape_increase"]
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]
    output_shape = [x + y for x, y in zip(shape_increase, net_config["output_shape"])]

    # prepare request
    voxel_size = gp.Coordinate(voxel_size)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    padding = None

    request = gp.BatchRequest()
    request.add(labels, input_size)
    request.add(obfuscated_labels, input_size)
    request.add(input_affs, input_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)

    # construct pipeline
    pipeline = CreateLabels(labels, shape=input_shape, voxel_size=voxel_size)
    pipeline += gp.Pad(labels, padding)
    pipeline += gp.SimpleAugment(transpose_only=[1, 2])
    pipeline += gp.DeformAugment(
        control_point_spacing=gp.Coordinate(4, 10, 10) * voxel_size,
        jitter_sigma=gp.Coordinate(1, 2, 2) * voxel_size,
        spatial_dims=3,
        subsample=1,
        scale_interval=(0.8, 1.2),
        rotation_axes=[1, 2],
    )
    pipeline += gp.ShiftAugment(prob_slip=0.1, prob_shift=0.1, sigma=3, p=0.8)

    if in_grow_boundary > 0:
        pipeline += CustomGrowBoundary(labels, max_steps=in_grow_boundary, only_xy=True)

    # introduce some errors in seg
    pipeline += ObfuscateLabels(labels, obfuscated_labels)

    # that is what predicted affs will look like
    pipeline += gp.AddAffinities(
        affinity_neighborhood=in_neighborhood,
        labels=obfuscated_labels,
        affinities=input_affs,
        dtype=np.float32,
    )

    # simulate noisy defected predictions
    pipeline += gp.NoiseAugment(input_affs, mode="gaussian", p=0.1)
    pipeline += gp.IntensityAugment(input_affs, 0.9, 1.1, -0.1, 0.1, slab=(1, -1, -1, -1), p=0.5)
    pipeline += gp.IntensityAugment(input_affs, 0.9, 1.1, -0.1, 0.1, slab=(-1, 1, -1, -1), p=0.5)
    pipeline += SmoothAugment(input_affs, slab=(-1, 1, -1, -1), blur_min=0.1, blur_max=1.5, p=0.5)
    pipeline += DefectAugment(
        input_affs, prob_low_contrast=0.1, prob_missing=0.0, prob_deform=0.0, axis=1
    )

    # now we erode - we want the gt affs to have a pixel boundary
    if out_aff_grow_boundary > 0:
        pipeline += gp.GrowBoundary(labels, steps=out_aff_grow_boundary, only_xy=True)

    pipeline += gp.AddAffinities(
        affinity_neighborhood=out_neighborhood,
        labels=labels,
        affinities=gt_affs,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights, slab=(3, -1, -1, -1))
    pipeline += gp.Stack(batch_size)

    return pipeline, request, keys


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AffsUNet()
        self.loss_fn = WeightedMSELoss()

    def forward(self, input_affs):
        return self.model(input_affs)

    def training_step(self, batch, batch_idx):
        pred_affs = self(batch["input_affs"])
        loss = self.loss_fn(pred_affs, batch["gt_affs"], batch["affs_weights"])
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "pred_affs": pred_affs.detach()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.5e-4)


def train(
    setup_dir,
    voxel_size,
    max_iterations,
    save_checkpoints_every,
    save_snapshots_every,
):
    # load net config
    with open(os.path.join(setup_dir, "net_config.json")) as f:
        net_config = json.load(f)

    # prepare dataset
    dataset = GunpowderDataset(create_pipeline, voxel_size, net_config)
    snapshot_callback = SnapshotCallback(setup_dir, voxel_size, save_snapshots_every)

    # train
    fit(
        LitModel(),
        dataset,
        setup_dir,
        max_iterations,
        save_checkpoints_every,
        snapshot_callback,
        num_workers=10,
    )


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = toml.load(f)

    assert config["setup_dir"] in setup_dir, "model directories do not match"
    config["setup_dir"] = setup_dir

    train(**config)
