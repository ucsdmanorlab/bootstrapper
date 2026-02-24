import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import gunpowder as gp
import zarr
import glob

from natsort import natsorted

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from model import AffsUNet, WeightedMSELoss
from bootstrapper.gp import (
    CreateLabels,
    Add2DLSDs,
    ObfuscateLabels,
    SmoothAugment,
    CustomGrowBoundary,
    DefectAugment, 
)

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
logging.getLogger().setLevel(logging.INFO)


def create_pipeline(voxel_size, net_config):
    """
    Returns Gunpowder pipeline, request, and keys.
    """
    batch_size = 1

    # array keys
    labels = gp.ArrayKey("SYNTHETIC_LABELS")
    obfuscated_labels = gp.ArrayKey("OBFUSCATED_LABELS")
    input_lsds = gp.ArrayKey("INPUT_2D_LSDS")
    input_affs = gp.ArrayKey("INPUT_2D_AFFS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

    keys = {
        'input_lsds': input_lsds,
        'input_affs': input_affs,
        'gt_affs': gt_affs,
        'affs_weights': affs_weights,
    }

    # get affs task params
    in_neighborhood = net_config["inputs"]["2d_affs"]["neighborhood"]
    in_neighborhood = [
        [0, *x] for x in in_neighborhood
    ]  # add z-dimension since pipeline is 3D
    in_grow_boundary = net_config["inputs"]["2d_affs"]["grow_boundary"]
    out_neighborhood = net_config["outputs"]["3d_affs"]["neighborhood"]
    out_aff_grow_boundary = net_config["outputs"]["3d_affs"]["grow_boundary"]

    # get lsd task params
    sigma = net_config["inputs"]["2d_lsds"]["sigma"]
    sigma = (0, sigma, sigma)  # add z-dimension since pipeline is 3D
    lsd_downsample = net_config["inputs"]["2d_lsds"]["downsample"]

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
    request.add(input_lsds, input_size)
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

    # that is what predicted lsds will look like
    pipeline += Add2DLSDs(obfuscated_labels, input_lsds, sigma=sigma, downsample=lsd_downsample)

    # that is what predicted affs will look like
    pipeline += gp.AddAffinities(
        affinity_neighborhood=in_neighborhood,
        labels=obfuscated_labels,
        affinities=input_affs,
        dtype=np.float32,
    )

    # simulate noisy defected predictions
    pipeline += gp.NoiseAugment(input_lsds, mode="gaussian", p=0.1)
    pipeline += gp.IntensityAugment(input_lsds, 0.9, 1.1, -0.1, 0.1, slab=(1, -1, -1, -1), p=0.5)
    pipeline += gp.IntensityAugment(input_lsds, 0.9, 1.1, -0.1, 0.1, slab=(-1, 1, -1, -1), p=0.5)
    pipeline += SmoothAugment(input_lsds, slab=(-1, 1, -1, -1), blur_min=0.1, blur_max=1.5, p=0.5)
    pipeline += DefectAugment(
        input_lsds, prob_low_contrast=0.1, prob_missing=0.0, prob_deform=0.0, axis=1
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


def gunpowder_generator(voxel_size, net_config):
    """
    Generator function that yields batches from gunpowder pipeline.
    """

    pipeline, request, keys = create_pipeline(
        voxel_size, net_config
    )
    
    with gp.build(pipeline):
        while True:
            try:
                batch = pipeline.request_batch(request)
                yield {k: torch.from_numpy(batch[v].data) for k,v in keys.items()}

            except Exception as e:
                logging.error(f"Error requesting batch: {e}")
                raise


class GunpowderDataset(IterableDataset):
    def __init__(self, generator, *args, **kwargs):
        super().__init__()
        self.generator = generator
        self.args = args
        self.kwargs = kwargs
        
    def __iter__(self):
        return self.generator(*self.args, **self.kwargs)


class SnapshotCallback(Callback):
    """
    PyTorch Lightning callback to save snapshots during training.
    """
    def __init__(self, voxel_size, setup_dir, save_every=100):
        super().__init__()
        self.voxel_size = voxel_size
        self.setup_dir = setup_dir
        self.save_every = save_every
        self.snapshots_dir = os.path.join(setup_dir, 'snapshots')
        os.makedirs(self.snapshots_dir, exist_ok=True)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the training batch ends."""
        
        # Only save every N steps
        if trainer.global_step != 1 and trainer.global_step % self.save_every != 0:
            return

        # # only save from rank 0
        # if trainer.global_rank != 0:
        #     return
        
        # Get predictions from the model
        with torch.no_grad():
            input_lsds = batch['input_lsds'].detach()
            input_affs = batch['input_affs'].detach()
            pred_affs = pl_module(input_lsds, input_affs).detach()
        
        # Prepare data for snapshot
        preds = {
            'pred_affs': pred_affs
        }

        name = f"batch_{trainer.global_step}_rank_{trainer.global_rank}.zarr"
        
        # Save snapshot
        self._save_snapshot(batch, preds, name)
    
    def _save_snapshot(self, batch, preds, name):
        """Save snapshot to zarr file."""
        
        snapshot_name = os.path.join(
            self.snapshots_dir,
            name
        )
        
        # Combine batch and predictions
        data = {**batch, **preds}
        
        # Calculate offset and voxel_size
        voxel_offset = (
            gp.Coordinate(data['input_lsds'].shape[-3:]) - 
            gp.Coordinate(data[list(preds.keys())[0]].shape[-3:])
        ) // 2
        voxel_size = gp.Coordinate(self.voxel_size)
        
        # Create zarr store
        store = zarr.DirectoryStore(snapshot_name)
        root = zarr.group(store=store, overwrite=True)
        
        for name, array in data.items():
            if isinstance(array, torch.Tensor):
                array = array.detach().cpu().numpy()
            
            root.create_dataset(
                name,
                data=array[0],
                overwrite=True
            )
            root[name].attrs['offset'] = (
                voxel_offset * voxel_size if (name != 'input_lsds' and name != 'input_affs') else [0, 0, 0]
            )
            root[name].attrs['voxel_size'] = voxel_size
        
        logging.info(f"Snapshot saved at: {os.path.abspath(snapshot_name)}")


class LitModel(pl.LightningModule):    
    def __init__(self):
        super().__init__()
        self.model = AffsUNet()
        self.loss_fn = WeightedMSELoss()
        
    def forward(self, input_lsds, input_affs):
        return self.model(input_lsds, input_affs)
    
    def training_step(self, batch, batch_idx):
        input_lsds = batch['input_lsds']
        input_affs = batch['input_affs']
        gt_affs = batch['gt_affs']
        affs_weights = batch['affs_weights']
        
        pred_affs = self(input_lsds, input_affs)
        
        loss = self.loss_fn(pred_affs, gt_affs, affs_weights)
        
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.5e-4)
        return optimizer    


def train(
    setup_dir=setup_dir,
    voxel_size=(1, 1, 1),
    max_iterations=25001,
    save_checkpoints_every=5000,
    save_snapshots_every=5000,
):
    # Load net config
    with open(os.path.join(setup_dir, "net_config.json")) as f:
        net_config = json.load(f)

    num_workers = 10
    
    # Create dataset
    dataset = GunpowderDataset(
        gunpowder_generator,
        voxel_size,
        net_config
    )
    
    # Create dataloader
    pl.seed_everything(42, workers=True)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
    )
    
    # Setup model, loss, optimizer
    model = LitModel()
    
    logger = pl.loggers.TensorBoardLogger(".", name="log", log_graph=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath='.', 
        filename='model_checkpoint_{step}',
        save_top_k=-1, 
        every_n_train_steps=save_checkpoints_every,
        auto_insert_metric_name=False,
    )
    snapshot_callback = SnapshotCallback(
        voxel_size=voxel_size,
        setup_dir=setup_dir,
        save_every=save_snapshots_every
    )

    trainer = pl.Trainer(
        max_steps=max_iterations,
        accelerator="gpu",
        devices=-1,
        benchmark=True,
        logger=logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, snapshot_callback]
    )

    latest_checkpoint = max(natsorted(glob.glob('model_*'))) if glob.glob('model_*') else None
    trainer.fit(model, dataloader, ckpt_path=latest_checkpoint)

if __name__ == "__main__":
    train()
