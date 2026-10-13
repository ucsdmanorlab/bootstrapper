import glob
import toml
import subprocess
import sys
import os
import click
import numpy as np
from funlib.persistence import open_ds

from .config import steps_of
import logging

SCHEMES = ("labels", "equal", "inv")


def setup_train(config_file, step=None, **kwargs):
    steps = steps_of(config_file, "train", step)
    if len(steps) > 1:
        names = ", ".join(s.name for s in steps)
        raise click.ClickException(f"{config_file} has several train steps ({names}); pass --step")
    config = dict(steps[0].keys)
    remake = f"bs prepare {os.path.dirname(os.path.abspath(config_file))}"
    if not os.path.exists(os.path.join(config["setup_dir"], "train.py")):
        raise click.ClickException(f"{config['setup_dir']} has no train.py; {remake} copies the model scripts there")

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
                fix = f"; {remake} makes it" if "/training_mask.zarr/" in mask else ""
                raise click.ClickException(f"training mask {mask} does not exist{fix}")
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
            check_inside("labels", labels_ds, "raw", raw)
            if mask_ds:
                check_inside("mask", mask_ds, "labels", labels_ds)
            out_samples.append(sample | {"labels": labels_ds, "mask": mask_ds})

    # update samples in config
    sampling = config.pop("sampling", "labels")
    if samples:
        for sample, p in zip(out_samples, probabilities(out_samples, sampling)):
            sample["probability"] = p
        if len(out_samples) > 1:
            click.echo("Sampling " + ", ".join(f"{_name(s)} {s['probability']:.0%}" for s in out_samples))
        config["samples"] = out_samples

    for key, value in kwargs.items():
        if value is not None:
            config[key] = value

    # the setup's train.py reads a flat file
    worker_config = os.path.join(config["setup_dir"], "train.toml")
    with open(worker_config, "w") as file:
        toml.dump(config, file)
    logging.info(f"Wrote {worker_config}")

    train_script = os.path.join(config["setup_dir"], "train.py")
    return train_script, worker_config


def probabilities(samples, sampling):
    """Each sample's share of the batches: by its labelled voxels ("labels"), equal ("equal"),
    by the inverse ("inv"), or your numbers, as a list in sample order or a table by volume."""
    if isinstance(sampling, str):
        if sampling not in SCHEMES:
            raise click.ClickException(f'sampling = "{sampling}" names no scheme; give "labels", "equal", "inv", or numbers such as [3, 1]')
        counts = [1] * len(samples)
        if sampling != "equal" and len(samples) > 1:
            counts = [labelled_voxels(s) for s in samples]
        weights = [1 / n if n else 0 for n in counts] if sampling == "inv" else counts
    else:
        weights = [sampling.get(_name(s)) for s in samples] if isinstance(sampling, dict) else sampling
        if len(weights) != len(samples) or None in weights:
            order = ", ".join(map(_name, samples))
            raise click.ClickException(f"sampling needs one number per volume, as a list in the order {order} or a table by name")
    return [w / sum(weights) for w in weights]


def _name(sample):
    return sample.get("volume", sample["labels"])


def labelled_voxels(sample):
    """The voxels a sample trains on: those of its training mask, else of its labels."""
    return int(np.count_nonzero(open_ds(sample["mask"] or sample["labels"]).data))


def check_inside(name, path, outer_name, outer_path):
    """gunpowder Pad stops training when a request reaches past the array it pads."""
    roi, outer = open_ds(path).roi, open_ds(outer_path).roi
    if not outer.contains(roi):
        crop = roi.intersect(outer)
        fix = f"crop the {name} to {crop} nm" if not crop.empty else "they do not overlap"
        raise click.ClickException(f"{name} {path} covers {roi} nm, past the {outer_name} at {outer} nm; {fix}")


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
    "--save-snapshots-every", type=int, help="Save snapshots every n iterations"
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
