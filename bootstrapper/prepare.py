import click
import os
import yaml
import logging
from pprint import pprint

from .configs import (
    save_config, 
    choose_model, 
    make_round_configs,
    create_training_config,
    create_prediction_configs,
    create_segmentation_configs,
    create_evaluation_configs,
    create_filter_configs,
    check_and_update
)
from .data.volumes import prepare_volume

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_volumes():
    """Prepare volumes for bootstrapping."""

    num_volumes = click.prompt(
        "Enter how many volumes to prepare?", default=1, type=int
    )

    volumes = []
    for i in range(num_volumes):
        logger.info(f"Processing volume {i+1}")
        volume_path = click.prompt(
            "Enter the path to the volume: ",
            type=click.Path(),
        )
        volume_info = prepare_volume(volume_path)
        if volume_info:
            volumes.append(volume_info)

    logger.info(f"Processed volumes: {volumes}")

    return volumes


def get_volumes():
    """ Get volumes from yaml file if exists, else ask for volumes info"""
    volumes = []

    if click.prompt("Enter if volumes.yaml already exists (y/n): ", default="n", type=click.Choice(["y", "n"])) == "y":
        volumes_yaml = click.prompt(
            "Enter the path to the volumes.yaml file: ",
            type=click.Path(exists=True, dir_okay=False, file_okay=True),
        )
        
        with open(volumes_yaml) as f:
            volumes = yaml.safe_load(f)
            logger.info(f"Loaded volumes from {volumes_yaml}: ")

    else:
        volumes = make_volumes()

    #save_config(volumes, volumes_yaml)
    return check_and_update(volumes)


def make_configs(base_dir):
    """Create for multiple rounds with given volumes."""

    existing_rounds = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir((os.path.join(base_dir, d)))
        and ".zarr" not in d
        and "round" in d
    ]

    logger.info(f"Existing rounds: {existing_rounds}")

    out_volumes = []
    i = 0

    while True:
        round_name = click.prompt(
            f"Enter name of round {i+1}: ", default=f"round_{i+1}"
        )
        round_dir = os.path.join(base_dir, round_name)
        os.makedirs(round_dir, exist_ok=True)

        if out_volumes == []:
            volumes = get_volumes()      
        else:
            volumes = out_volumes

        logger.info(f"Writing volumes to {round_dir}/volumes.yaml")
        save_config(volumes, os.path.join(round_dir, "volumes.yaml"))  

        model_name = choose_model()
        setup_dir = os.path.join(round_dir, model_name)
        out_volumes = make_round_configs(volumes, setup_dir, model_name)

        if click.prompt(
            "Make configs for next round? (y/n): ",
            type=click.Choice(["y", "n"]),
            default="n") == "y":
                i += 1
        else:
            break
        
    logger.info(f"All configs created successfully!")


class OrderedGroup(click.Group):
    def list_commands(self, ctx):
        return [
            "volumes",
            "train",
            "predict",
            "segment",
            "eval",
            "filter",
        ]


@click.group(invoke_without_command=True, cls=OrderedGroup)
@click.pass_context
def prepare(ctx):
    """
    Prepare volumes and config files for bootstrapping

    This command sets up a pipeline for training, prediction, post-processing,
    evaluation, and filtering segmentations across multiple rounds. It handles:

    - Volume preparation: Process raw and label data (tif/zarr)

    - Configuration generation for each round:

        - Training: Set model and parameters

        - Prediction: Configure for each volume

        - Segmentation: Set up RAG, blockwise for each volume

        - Evaluation: Configure lsd error metrics and ground truth comparisons

        - Filtering: Set up pseudo ground truth generation

    The process is iterative, with each round building upon previous results,
    allowing for refinement of the segmentations over time.
    """
    if ctx.invoked_subcommand is None:

        base_dir = click.prompt(
            "Enter the path to the base directory: ",
            type=click.Path(),
            default=os.getcwd(),
        )
        os.makedirs(base_dir, exist_ok=True)

        make_configs(base_dir)
    pass


@prepare.command('volumes')
def prep_vol():
    """Prepare a single volume for bootstrapping."""
    volumes = make_volumes()
    if click.prompt(
        "Enter whether to save volumes.yaml (y/n): ",
        default="y",
        type=click.Choice(["y", "n"]),
        show_default=True,
    ) == "y":
        volumes_yaml = click.prompt(
            "Enter the path to new volumes.yaml file (enter to skip): ",
            type=click.Path(),
            default=os.path.join(os.getcwd(), "volumes.yaml"),
        )
        save_config(volumes, volumes_yaml)


@prepare.command('train')
def prep_train_config():
    """Create training config files."""
    setup_dir = click.prompt(
        "Enter the path to the setup directory to create: ",
        type=click.Path(),
    )
    model_name = choose_model()
    volumes = get_volumes()

    ret = create_training_config(volumes, setup_dir, model_name)

    config_path = click.prompt(
        "Enter the path to save the train config file: ",
        type=click.Path(),
        default=os.path.join(setup_dir, "run", "train.yaml")
    )
    save_config(ret, config_path)


@prepare.command('predict')
def prep_predict_config():
    """Create prediction config files."""
    volumes = get_volumes()
    setup_dir = click.prompt(
        "Enter the path to the setup directory: ",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
    )

    ret = create_prediction_configs(volumes, setup_dir)

    for volume_name, config in ret["configs"].items():
        config_path = click.prompt(
            f"Enter the path to save the predict config file for {volume_name}: ",
            type=click.Path(),
            default=os.path.join(setup_dir, "run", f"predict_{volume_name}.yaml")
        )
        save_config(config, config_path)


@prepare.command('segment')
def prep_segment_config():
    """Create segmentation config files."""
    volumes = get_volumes()
    setup_dir = click.prompt(
        "Enter the path to the setup directory: ",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
    )
    out_affs_ds = click.prompt(
        "Enter the name of the output affinities dataset: ",
        type=str,
    )
    
    ret = create_segmentation_configs(volumes, setup_dir, out_affs_ds)

    for volume_name, config in ret["configs"].items():
        config_path = click.prompt(
            f"Enter the path to save the segment config file for {volume_name}: ",
            type=click.Path(),
            default=os.path.join(setup_dir, "run", f"seg_{volume_name}.yaml"),
        )
        save_config(config, config_path)


@prepare.command('eval')
def prep_eval_config():
    """Create evaluation config files."""
    volumes = get_volumes()
    setup_dir = click.prompt(
        "Enter the path to the setup directory: ",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
    )
    out_segs_prefix = click.prompt(
        "Enter the prefix for the segmentation datasets: ",
        type=str,
    )
    pred_datasets = click.prompt(
        "Enter the prediction datasets (commma-separated): ",
        type=str,
    )
    pred_datasets = [x.strip() for x in pred_datasets.split(",")]

    ret = create_evaluation_configs(volumes, setup_dir, out_segs_prefix, pred_datasets)

    for volume_name, config in ret["configs"].items():
        config_path = click.prompt(
            f"Enter the path to save the eval config file for {volume_name}: ",
            type=click.Path(),
            default=os.path.join(setup_dir, "run", f"eval_{volume_name}.yaml"),
        )
        save_config(config, config_path)


@prepare.command('filter')
def prep_filter_config():
    """Create config files for filtering segmentations."""
    volumes = get_volumes()
    setup_dir = click.prompt(
        "Enter the path to the setup directory: ",
        type=click.Path(exists=True, dir_okay=True, file_okay=False),
    )
    out_segs_prefix = click.prompt(
        "Enter the prefix for the segmentation datasets: ",
        type=str,
    )
    eval_dir = click.prompt(
        "Enter the path to the evaluation directory: ",
        type=click.Path(),
    )

    ret = create_filter_configs(volumes, setup_dir, out_segs_prefix, eval_dir)

    for volume_name, config in ret["configs"].items():
        config_path = click.prompt(
            f"Enter the path to save the filter config file for {volume_name}: ",
            type=click.Path(),
            default=os.path.join(setup_dir, "run", f"filter_{volume_name}.yaml"),
        )
        save_config(config, config_path)