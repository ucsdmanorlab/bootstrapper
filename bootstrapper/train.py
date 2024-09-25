import argparse
import yaml
import subprocess
import os
import click
import logging

logging.basicConfig(level=logging.INFO)


def extract_setup_dir(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config["setup_dir"]


def run_training(yaml_file, **kwargs):
    setup_dir = extract_setup_dir(yaml_file)
    train_script = os.path.join(setup_dir, "train.py")

    # Load the config file
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    config_file = yaml_file
    if any(kwargs.values()):
        # Override config values with provided kwargs
        for key, value in kwargs.items():
            if value is not None:
                config[key] = value

        # Write the updated config to a temporary file
        config_file = os.path.join(os.path.dirname(yaml_file), "temp_train.yaml")
        counter = 1
        while os.path.exists(config_file):
            config_file = os.path.join(
                os.path.dirname(yaml_file), f"temp_train_{counter}.yaml"
            )
            counter += 1
        with open(config_file, "w") as file:
            yaml.dump(config, file)

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
@click.argument("yaml_file", type=click.Path(exists=True))
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
    yaml_file,
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
        yaml_file,
        max_iterations=max_iterations,
        output_dir=output_dir,
        save_checkpoints_every=save_checkpoints_every,
        save_snapshots_every=save_snapshots_every,
        voxel_size=voxel_size,
        sigma=sigma,
    )
