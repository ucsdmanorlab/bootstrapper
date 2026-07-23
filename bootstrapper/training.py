import glob
import logging
import os

import gunpowder as gp
import numpy as np
import pytorch_lightning as pl
import torch
import zarr
from natsort import natsorted
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, RichProgressBar
from torch.utils.data import DataLoader, IterableDataset


class GunpowderDataset(IterableDataset):
    """Infinite dataset over a gunpowder pipeline; each dataloader worker
    builds and requests from its own copy of the pipeline."""

    def __init__(self, create_pipeline, *args, **kwargs):
        super().__init__()
        self.create_pipeline = create_pipeline
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        pipeline, request, keys = self.create_pipeline(*self.args, **self.kwargs)
        with gp.build(pipeline):
            while True:
                batch = pipeline.request_batch(request)
                yield {k: torch.from_numpy(batch[v].data) for k, v in keys.items()}


class StepProgressBar(RichProgressBar):
    """Show progress against max_steps instead of the unknown per-epoch batch
    count of the infinite dataset (which renders as '/--')."""

    def _get_train_description(self, current_epoch):
        return "Training"

    @property
    def total_train_batches(self):
        max_steps = self.trainer.max_steps
        return max_steps if max_steps and max_steps > 0 else super().total_train_batches


class SnapshotCallback(Callback):
    """Save a zarr snapshot of the batch and the model's predictions every
    `save_every` steps. Float arrays scaled to [-1, 1] are saved back as
    [0, 255] uint8; everything else is saved as-is."""

    def __init__(self, setup_dir, voxel_size, save_every):
        super().__init__()
        self.voxel_size = gp.Coordinate(voxel_size)
        self.save_every = save_every
        self.snapshots_dir = os.path.join(setup_dir, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step != 1 and step % self.save_every != 0:
            return

        # combine batch and predictions
        data = dict(batch)
        data.update({k: v for k, v in outputs.items() if k != "loss"})

        self._save_snapshot(data, f"batch_{step}_rank_{trainer.global_rank}.zarr")

    def _save_snapshot(self, data, name):
        path = os.path.join(self.snapshots_dir, name)
        n = len(self.voxel_size)

        # output-sized arrays are centered inside the input-sized (largest) ones
        shapes = [gp.Coordinate(a.shape[-n:]) for a in data.values()]
        in_shape = gp.Coordinate(max(s[d] for s in shapes) for d in range(n))

        store = zarr.DirectoryStore(path)
        root = zarr.group(store=store, overwrite=True)

        for key, array in data.items():
            array = array.detach().cpu().numpy()

            # save [-1, 1] floats as uint8
            if not np.issubdtype(array.dtype, np.integer):
                lo, hi = array.min(), array.max()
                if -1 <= lo < 0 and hi <= 1:
                    array = ((array * 0.5 + 0.5) * 255).astype(np.uint8)
            offset = (in_shape - gp.Coordinate(array.shape[-n:])) // 2
            root.create_dataset(key, data=array, overwrite=True)
            root[key].attrs["offset"] = list(offset * self.voxel_size)
            root[key].attrs["voxel_size"] = list(self.voxel_size)

        logging.info(f"Snapshot saved at: {os.path.abspath(path)}")


def fit(
    model,
    dataset,
    setup_dir,
    max_iterations,
    save_checkpoints_every,
    snapshot_callback,
    num_workers=8,
):
    pl.seed_everything(42, workers=True)

    # prepare dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    # prepare trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=setup_dir,
        filename="model_checkpoint_{step}",
        save_top_k=-1,
        every_n_train_steps=save_checkpoints_every,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_steps=max_iterations,
        max_epochs=1,
        use_distributed_sampler=False,
        benchmark=True,
        logger=pl.loggers.TensorBoardLogger(setup_dir, name="log"),
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, snapshot_callback, StepProgressBar()],
    )

    # resume from latest checkpoint
    ckpts = natsorted(glob.glob(os.path.join(setup_dir, "model_checkpoint_*.ckpt")))
    trainer.fit(model, dataloader, ckpt_path=ckpts[-1] if ckpts else None)
