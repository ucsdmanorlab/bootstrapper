import glob
import toml
import subprocess
import os
import click
import logging

logging.basicConfig(level=logging.INFO)


def setup_train(config_file, **kwargs):
    with open(config_file, "r") as file:
        config = toml.load(file)

    # get training samples
    samples = config["samples"]
    if not samples:
        raise ValueError(f"No training samples provided in {config_file}")

    # check training samples
    out_samples = []
    for sample in samples:
        raw = sample["raw"]
        labels = sample["labels"]
        mask = sample["mask"]

        # check raw
        if not os.path.exists(raw):
            raise ValueError(f"Raw dataset path {raw} does not exist")
        elif ".zarray" not in os.listdir(raw):
            raise ValueError(f"Raw dataset path {raw} does not contain a zarr array")

        # check labels, find all contained arrays if just a prefix
        if not os.path.exists(labels):
            raise ValueError(f"Labels dataset path {labels} does not exist")
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
                raise ValueError(f"Labels dataset path {labels} does not exist")
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
    config["samples"] = out_samples

    # Override config values with provided kwargs
    config_file = config_file
    if any(kwargs.values()):
        for key, value in kwargs.items():
            if value is not None:
                config[key] = value

        base_name = config_file.replace(".toml", "_modified.toml")
        counter = 0

        # write updated config
        while True:
            config_file = f"{base_name}_{counter}.toml"
            if not os.path.exists(config_file):
                break
            counter += 1

    # write updated config
    logging.info(f"Using updated config {config_file}")
    with open(config_file, "w") as file:
        toml.dump(config, file)

    train_script = os.path.join(config["setup_dir"], "train.py")
    return train_script, config_file


def run_training(config_file, **kwargs):

    train_script, config_file = setup_train(config_file, **kwargs)

    # Run the training script with the temporary config file
    command = ["python", train_script, config_file]
    logging.info(f"Starting training with command: {' '.join(command)}")

    try:
        result = subprocess.run(command)
        logging.info("Training completed successfully.")
        logging.debug(f"Training output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed with error code {e.returncode}")
        logging.error(f"Error output:\n{e.stderr}")


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--max_iterations", "-i", type=int, help="Number of training iterations")
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Output directory",
)
@click.option(
    "--save_checkpoints_every",
    "-ce",
    type=int,
    help="Save checkpoints every n iterations",
)
@click.option(
    "--save_snapshots_every", "-s", type=int, help="Save snapshots every n iterations"
)
@click.option(
    "--voxel_size", "-v", type=str, help="Voxel size (space-separated integers)"
)
@click.option("--sigma", "-s", type=int, help="Sigma value for LSD models")
def train(
    config_file,
    max_iterations,
    output_dir,
    save_checkpoints_every,
    save_snapshots_every,
    voxel_size,
    sigma,
):
    """
    Run training with the specified config file.

    Optional parameters will override the corresponding values in the config file.
    """

    if voxel_size:
        voxel_size = [int(v) for v in voxel_size.strip().split()]

    run_training(
        config_file,
        max_iterations=max_iterations,
        output_dir=output_dir,
        save_checkpoints_every=save_checkpoints_every,
        save_snapshots_every=save_snapshots_every,
        voxel_size=voxel_size,
        sigma=sigma,
    )
