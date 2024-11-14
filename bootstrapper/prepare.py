import click
import os
import toml

from .configs import (
    save_config,
    download_checkpoints,
    make_round_configs,
    create_training_config,
    create_prediction_configs,
    create_segmentation_configs,
    create_evaluation_configs,
    create_filter_configs,
    check_and_update,
)
from .data.volumes import prepare_volume
from .styles import cli_echo, cli_prompt, cli_confirm


def make_volumes(round_dir=None, style="prepare"):
    """Prepare volumes for bootstrapping."""

    if round_dir is None:
        round_dir = os.getcwd()
    else:
        assert os.path.isdir(round_dir)

    num_volumes = cli_prompt(
        f"Enter number of volumes to prepare in {round_dir}",
        style,
        default=1,
        type=int,
    )

    volumes = {}
    for i in range(num_volumes):
        click.echo()
        cli_echo(f"Processing volume {i+1}", style)
        vol_name = cli_prompt(
            f"Enter name of volume {i+1} in {round_dir}",
            style,
            default=f"volume_{i+1}",
        )
        volume_info = prepare_volume(os.path.join(round_dir, f"{vol_name}.zarr"))
        if volume_info:
            volumes[vol_name] = volume_info

    return volumes


def get_volumes(round_dir=None, style="prepare"):
    """Get volumes from config file if exists, else ask for volumes info"""

    if round_dir is not None and os.path.exists(
        os.path.join(round_dir, "volumes.toml")
    ):
        volumes_doc = os.path.abspath(os.path.join(round_dir, "volumes.toml"))
        load_volumes = cli_confirm(
            f"Load volumes from {volumes_doc}?", style, default=True
        )
    elif os.path.exists(os.path.join(os.getcwd(), "volumes.toml")):
        volumes_doc = os.path.abspath(os.path.join(os.getcwd(), "volumes.toml"))
        load_volumes = cli_confirm(
            f"Load volumes from {volumes_doc}?", style, default=True
        )
    else:
        load_volumes = False

    if load_volumes:
        with open(volumes_doc) as f:
            volumes = toml.load(f)
            cli_echo(f"Loaded volumes from {volumes_doc}", style)
    else:
        volumes = make_volumes(round_dir)

    return check_and_update(volumes, style)


def make_configs(base_dir):
    """Create configs for multiple rounds."""

    existing_rounds = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir((os.path.join(base_dir, d)))
        and ".zarr" not in d
        and "round" in d
    ]

    cli_echo(f"Existing rounds: {existing_rounds}", style="prepare")

    out_volumes = {}
    i = 0

    while True:
        click.echo()
        round_name = cli_prompt(
            f"Enter name for round {i+1}",
            style="prepare",
            default=f"round_{i+1}",
        )
        round_dir = os.path.join(base_dir, round_name)
        os.makedirs(round_dir, exist_ok=True)

        if not out_volumes:
            volumes = get_volumes(round_dir=round_dir)
        else:
            volumes = {
                vol_name: vol_info
                | {
                    "output_container": os.path.join(
                        round_dir, f"{vol_info['name']}.zarr"
                    )
                }
                for vol_name, vol_info in out_volumes.items()
            }

        cli_echo(f"Writing volumes to {round_dir}/volumes.toml", style="prepare")
        save_config(volumes, os.path.join(round_dir, "volumes.toml"), style="prepare")

        out_volumes = make_round_configs(volumes, round_dir)

        click.echo()
        if not cli_confirm(
            "Make configs for next round?", style="prepare", default=False
        ):
            break
        i += 1

    click.echo()
    cli_echo("All configs created successfully!", style="prepare", stype="success")


class PrepareGroup(click.Group):
    def list_commands(self, ctx):
        return [
            "volumes",
            "train",
            "predict",
            "segment",
            "eval",
            "filter",
        ]

    def get_command(self, ctx, cmd_name):
        ret = click.Group.get_command(self, ctx, cmd_name)
        if ret is not None:
            return ret

        aliases = {
            "vols": "volumes",
            "vol": "volumes",
            "v": "volumes",
            "train": "train",
            "t": "train",
            "pred": "predict",
            "p": "predict",
            "seg": "segment",
            "s": "segment",
            "eval": "evaluate",
            "e": "eval",
            "f": "filter",
        }

        if cmd_name in aliases:
            return click.Group.get_command(self, ctx, aliases[cmd_name])
        return None


@click.group(invoke_without_command=True, cls=PrepareGroup, chain=True)
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
        base_dir = cli_prompt(
            "Enter path to base directory",
            style="prepare",
            type=click.Path(),
            default=os.getcwd(),
        )
        os.makedirs(base_dir, exist_ok=True)

        make_configs(base_dir)
    pass


@prepare.command("volumes")
def prep_vol():
    """Prepare a single volume."""
    volumes = make_volumes()
    click.echo()
    if cli_confirm("Save volumes.toml?", default=True):
        volumes_doc = cli_prompt(
            "Enter path for new volumes.config file",
            type=click.Path(),
            default=os.path.join(os.getcwd(), "volumes.toml"),
        )
        save_config(volumes, volumes_doc)


@prepare.command("train")
def prep_train_config():
    """Create training config files."""

    volumes = get_volumes(style="train")
    ret = create_training_config(volumes, os.getcwd())

    click.echo()
    for setup_dir in ret["configs"]:
        config_path = cli_prompt(
            f"Enter path to save train config file for {setup_dir}",
            style="train",
            type=click.Path(),
            default=os.path.join(
                os.getcwd(), f"train_{os.path.basename(setup_dir)}.toml"
            ),
        )
        save_config(ret["configs"][setup_dir], config_path, style="train")


@prepare.command("predict")
def prep_predict_config():
    """Create prediction config files."""
    volumes = get_volumes(style="predict")

    # get list of setup directories
    setup_dirs = cli_prompt(
        f"Enter setups paths for prediction in order (comma-separated)", style="predict"
    )
    setup_dirs = [x.strip() for x in setup_dirs.split(",")]

    # check if all setup directories exist
    for i, setup_dir in enumerate(setup_dirs):
        if not os.path.isdir(setup_dir):
            # accept just names of '_from_' models. for example, '3d_affs_from_2d_affs'
            if "_from_" in setup_dir:
                setup_dir = os.path.join(os.path.dirname(__file__), "models", setup_dir)
                if not os.path.isdir(setup_dir):
                    raise ValueError(
                        f"Invalid setup: directory {setup_dir} does not exist"
                    )

                checkpoints = [
                    c for c in os.listdir(setup_dir) if "model_checkpoint_" in c
                ]
                if not checkpoints:
                    cli_echo(f"No checkpoints found in {setup_dir}", style="predict")
                    download = cli_confirm(
                        f"Download pretrained checkpoints for {os.path.basename(setup_dir)}?",
                        style="predict",
                        default=False,
                    )
                    if download:
                        download_checkpoints(
                            os.path.basename(setup_dir), setup_dir, style="predict"
                        )
                    else:
                        raise ValueError(
                            f"Please either download checkpoints or train from scratch"
                        )

                # replace setup_dir with setup_dir from bootstrapper source directory
                setup_dirs[i] = setup_dir

            else:
                raise ValueError(f"Invalid setup: directory {setup_dir} does not exist")
        else:
            setup_dirs[i] = os.path.abspath(setup_dir)

    ret = create_prediction_configs(volumes, setup_dirs)

    for volume_name, config in ret["configs"].items():
        click.echo()
        config_path = cli_prompt(
            f"Enter path to save predict config for {volume_name}",
            style="predict",
            type=click.Path(),
            default=os.path.join(os.getcwd(), f"predict_{volume_name}.toml"),
        )
        save_config(config, config_path, style="predict")


@prepare.command("segment")
def prep_segment_config():
    """Create segmentation config files."""
    volumes = get_volumes(style="segment")
    out_affs_ds = cli_prompt(
        "Enter name of output affinities dataset inside zarr", style="segment"
    )
    ret = create_segmentation_configs(volumes, out_affs_ds)

    for volume_name, config in ret["configs"].items():
        click.echo()
        config_path = cli_prompt(
            f"Enter path to save segment config for {volume_name}",
            style="segment",
            type=click.Path(),
            default=os.path.join(os.getcwd(), f"seg_{volume_name}.toml"),
        )
        save_config(config, config_path, style="segment")


@prepare.command("evaluate")
def prep_eval_config():
    """Create evaluation config files."""
    volumes = get_volumes(style="evaluate")
    out_segs_prefix = cli_prompt(
        "Enter prefix for segmentation datasets", style="evaluate"
    )
    pred_datasets = cli_prompt(
        "Enter prediction datasets (comma-separated)", style="evaluate"
    )
    pred_datasets = [x.strip() for x in pred_datasets.split(",")]

    ret = create_evaluation_configs(volumes, out_segs_prefix, pred_datasets)

    for volume_name, config in ret["configs"].items():
        click.echo()
        config_path = cli_prompt(
            "Enter path to save eval config for {volume_name}",
            style="evaluate",
            type=click.Path(),
            default=os.path.join(os.getcwd(), f"eval_{volume_name}.toml"),
        )
        save_config(config, config_path, style="evaluate")


@prepare.command("filter")
def prep_filter_config():
    """Create config files for filtering segmentations."""
    volumes = get_volumes(style="filter")
    out_segs_prefix = cli_prompt(
        "Enter prefix for segmentation datasets", style="filter"
    )
    eval_dir = cli_prompt("Enter path to evaluation directory", style="filter")
    ret = create_filter_configs(volumes, out_segs_prefix, eval_dir)

    for volume_name, config in ret["configs"].items():
        click.echo()
        config_path = cli_prompt(
            f"Enter path to save filter config for {volume_name}",
            style="filter",
            type=click.Path(),
            default=os.path.join(os.getcwd(), f"filter_{volume_name}.toml"),
        )
        save_config(config, config_path, style="filter")
