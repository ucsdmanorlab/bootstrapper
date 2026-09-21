import glob
import toml
import subprocess
import sys
import os
import click

from .config import steps_of
import logging

logging.basicConfig(level=logging.INFO)


def setup_train(config_file, step=None, **kwargs):
    _, steps = steps_of(config_file, "train", step)
    if len(steps) > 1:
        names = ", ".join(s.name for s in steps)
        raise click.ClickException(f"{config_file} has several train steps ({names}); pass --step")
    config = dict(steps[0].keys)

    # get training samples (synthetic *_from_* setups have none)
    samples = config.get("samples", [])
    if "samples" in config and not samples:
        raise ValueError(f"No training samples provided in {config_file}")

    # check training samples
    out_samples = []
    for sample in samples:
        raw = sample["raw"]
        labels = sample["labels"]
        mask = None if "mask" not in sample else sample["mask"]

        # check raw
        if not os.path.exists(raw):
            raise ValueError(f"Raw dataset path {raw} does not exist")
        elif ".zarray" not in os.listdir(raw):
            raise ValueError(f"Raw dataset path {raw} does not contain a zarr array")

        # check labels, find all contained arrays if just a prefix
        if not os.path.exists(labels):
            raise ValueError(
                f"Labels dataset path {labels} does not exist:"
                "point the config at an existing labels dataset"
            )
        elif ".zarray" not in os.listdir(labels):
            # recursively search for all arrays matching the prefix
            labels_datasets = [
                os.path.dirname(x)
                for x in glob.glob(
                    os.path.join(labels, "**", ".zarray"), recursive=True
                )
            ]
            if len(labels_datasets) == 0:
                raise ValueError(
                    f"Labels dataset prefix {labels} does not contain any array"
                )
        else:
            labels_datasets = [labels]

        # check mask, find all contained arrays if just a prefix and not None
        if mask is not None:
            if not os.path.exists(mask):
                raise ValueError(f"Mask dataset path {mask} does not exist")
            elif ".zarray" not in os.listdir(mask):
                # recursively search for all arrays matching the prefix
                mask_datasets = [
                    os.path.dirname(x)
                    for x in glob.glob(
                        os.path.join(mask, "**", ".zarray"), recursive=True
                    )
                ]
                if len(mask_datasets) == 0:
                    raise ValueError(
                        f"Mask dataset prefix {mask} does not contain any array"
                    )
            else:
                mask_datasets = [mask]
        else:
            mask_datasets = [None for _ in labels_datasets]

        assert len(labels_datasets) == len(
            mask_datasets
        ), "Number of labels and mask datasets must be equal"

        # update sample
        for labels_ds, mask_ds in zip(labels_datasets, mask_datasets):
            out_samples.append(
                {
                    "raw": raw,
                    "labels": labels_ds,
                    "mask": mask_ds,
                }
            )

    # update samples in config
    if samples:
        config["samples"] = out_samples

    for key, value in kwargs.items():
        if value is not None:
            config[key] = value

    # the setup's train.py reads a flat file; it lives beside the checkpoints it makes
    worker_config = os.path.join(config["setup_dir"], "train.toml")
    with open(worker_config, "w") as file:
        toml.dump(config, file)
    logging.info(f"Wrote {worker_config}")

    train_script = os.path.join(config["setup_dir"], "train.py")
    return train_script, worker_config


def run_training(config_file, step=None, **kwargs):

    train_script, config_file = setup_train(config_file, step, **kwargs)

    # Run the training script with the temporary config file
    command = [sys.executable, train_script, config_file]
    logging.info(f"Starting training with command: {' '.join(command)}")

    subprocess.run(command, check=True)
    logging.info("Training completed successfully.")


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--step", "-s", type=str, help="The train step to run when the file has several")
@click.option("--max-iterations", "-n", type=int, help="Number of training iterations")
@click.option(
    "--save-checkpoints-every",
    "-ce",
    type=int,
    help="Save checkpoints every n iterations",
)
@click.option(
    "--save-snapshots-every", "-s", type=int, help="Save snapshots every n iterations"
)
@click.option(
    "--voxel-size", "-v", type=str, help="Voxel size (space-separated integers)"
)
def train(
    config_file,
    step,
    max_iterations,
    save_checkpoints_every,
    save_snapshots_every,
    voxel_size,
):
    """
    Run training with the specified config file.

    Optional parameters will override the corresponding values in the config file.
    """

    if voxel_size:
        voxel_size = [int(v) for v in voxel_size.strip().split()]

    run_training(
        config_file,
        step,
        max_iterations=max_iterations,
        save_checkpoints_every=save_checkpoints_every,
        save_snapshots_every=save_snapshots_every,
        voxel_size=voxel_size,
    )
