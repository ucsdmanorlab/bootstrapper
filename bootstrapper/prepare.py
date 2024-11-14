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


def make_volumes(round_dir=None):
    """Prepare volumes for bootstrapping."""

    if round_dir is None:
        round_dir = os.getcwd()
    else:
        assert os.path.isdir(round_dir)

    num_volumes = click.prompt(
        click.style(f"Enter number of volumes to prepare in {round_dir}", fg="cyan"),
        default=1,
        type=int,
    )

    volumes = []
    for i in range(num_volumes):
        click.echo()
        click.secho(f"Processing volume {i+1}", fg="cyan", bold=True)
        vol_name = click.prompt(
            click.style(f"Enter name of volume {i+1} in {round_dir}", fg="cyan"),
            type=str,
            default=f"volume_{i+1}",
        )
        volume_info = prepare_volume(os.path.join(round_dir, f"{vol_name}.zarr"))
        if volume_info:
            volumes.append(volume_info)

    return volumes


def get_volumes(round_dir=None):
    """Get volumes from config file if exists, else ask for volumes info"""
    volumes = []

    if round_dir is not None and os.path.exists(os.path.join(round_dir, "volumes.toml")):
        volume_list = os.path.abspath(os.path.join(round_dir, "volumes.toml"))
        load_volumes = click.confirm(
            click.style(f"Load volumes from {volume_list}?", fg="cyan"), default=True
        )
    elif os.path.exists(os.path.join(os.getcwd(), "volumes.toml")):
        volume_list = os.path.abspath(os.path.join(os.getcwd(), "volumes.toml"))
        load_volumes = click.confirm(
            click.style(f"Load volumes from {volume_list}?", fg="cyan"), default=True
        )
    else:
        load_volumes = False

    if load_volumes:
        with open(volume_list) as f:
            volumes = toml.load(f)
            click.secho(f"Loaded volumes from {volume_list}", fg="cyan", bold=True)
    else:
        volumes = make_volumes(round_dir)

    return check_and_update(volumes)


def make_configs(base_dir):
    """Create configs for multiple rounds."""

    existing_rounds = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir((os.path.join(base_dir, d)))
        and ".zarr" not in d
        and "round" in d
    ]

    click.secho(f"Existing rounds: {existing_rounds}", fg="cyan", bold=True)

    out_volumes = []
    i = 0

    while True:
        click.echo()
        round_name = click.prompt(
            click.style(f"Enter name for round {i+1}", fg="cyan"),
            default=f"round_{i+1}",
        )
        round_dir = os.path.join(base_dir, round_name)
        os.makedirs(round_dir, exist_ok=True)

        if not out_volumes:
            volumes = get_volumes(round_dir=round_dir)
        else:
            volumes = [
                volume | {"output_container": os.path.join(round_dir, f"{volume["name"]}.zarr")}
                for volume in out_volumes
            ]

        click.secho(
            f"Writing volumes to {round_dir}/volumes.toml", fg="cyan", bold=True
        )
        save_config(volumes, os.path.join(round_dir, "volumes.toml"))

        out_volumes = make_round_configs(volumes, round_dir)

        click.echo()
        if not click.confirm(
            click.style("Make configs for next round?", fg="cyan"), default=False
        ):
            break
        i += 1

    click.echo()
    click.secho("All configs created successfully!", fg="cyan", bold=True)


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
            'vols': 'volumes',
            'vol': 'volumes',
            'v': 'volumes',
            'train': 'train',
            't': 'train',
            'pred': 'predict',
            'p': 'predict',
            'seg': 'segment',
            's': 'segment',
            'eval': 'evaluate',
            'e': 'eval',
            'f': 'filter',
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
        base_dir = click.prompt(
            click.style("Enter path to base directory", fg="cyan"),
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
    if click.confirm(click.style("Save volumes.toml?", fg="cyan"), default=True):
        volume_list = click.prompt(
            click.style("Enter path for new volumes.config file", fg="cyan"),
            type=click.Path(),
            default=os.path.join(os.getcwd(), "volumes.toml"),
        )
        save_config(volumes, volume_list)


@prepare.command("train")
def prep_train_config():
    """Create training config files."""

    volumes = get_volumes()
    ret = create_training_config(volumes, os.getcwd())

    click.echo()
    for setup_dir in ret["configs"]:
        config_path = click.prompt(
            click.style(f"Enter path to save train config file for {setup_dir}", fg="cyan"),
            type=click.Path(),
            default=os.path.join(os.getcwd(), f"train_{os.path.basename(setup_dir)}.toml"),
        )
        save_config(ret["configs"][setup_dir], config_path)


@prepare.command("predict")
def prep_predict_config():
    """Create prediction config files."""
    volumes = get_volumes()

    # get list of setup directories
    setup_dirs = click.prompt(
        click.style(f"Enter setups paths for prediction in order (comma-separated)", fg="cyan"),
    )
    setup_dirs = [x.strip() for x in setup_dirs.split(',')]

    # check if all setup directories exist
    for i, setup_dir in enumerate(setup_dirs):
        if not os.path.isdir(setup_dir):
            # accept just names of '_from_' models. for example, '3d_affs_from_2d_affs' 
            if '_from_' in setup_dir:
                setup_dir = os.path.join(os.path.dirname(__file__), "models", setup_dir)
                if not os.path.isdir(setup_dir):
                    raise ValueError(f"Invalid setup: directory {setup_dir} does not exist")
                
                checkpoints = [
                    c for c in os.listdir(setup_dir) if 'model_checkpoint_' in c
                ]
                if not checkpoints:
                    click.secho(f"No checkpoints found in {setup_dir}", fg='cyan')
                    download = click.confirm(
                        click.style(f"Download pretrained checkpoints for {os.path.basename(setup_dir)}?", fg='cyan'),
                        default=False
                    )
                    if download:
                        download_checkpoints(os.path.basename(setup_dir), setup_dir)
                    else:
                        raise ValueError(f"Please either download checkpoints or train from scratch")
                
                # replace setup_dir with setup_dir from bootstrapper source directory
                setup_dirs[i] = setup_dir

            else:
                raise ValueError(f"Invalid setup: directory {setup_dir} does not exist")
        else:
            setup_dirs[i] = os.path.abspath(setup_dir)

    ret = create_prediction_configs(volumes, setup_dirs)

    for volume_name, config in ret["configs"].items():
        click.echo()
        config_path = click.prompt(
            click.style(
                f"Enter path to save predict config for {volume_name}", fg="cyan"
            ),
            type=click.Path(),
            default=os.path.join(os.getcwd(), f"predict_{volume_name}.toml"),
        )
        save_config(config, config_path)


@prepare.command("segment")
def prep_segment_config():
    """Create segmentation config files."""
    volumes = get_volumes()

    out_affs_ds = click.prompt(
        click.style("Enter name of output affinities dataset inside zarr", fg="cyan"),
        type=str,
    )

    ret = create_segmentation_configs(volumes, out_affs_ds)

    for volume_name, config in ret["configs"].items():
        click.echo()
        config_path = click.prompt(
            click.style(
                f"Enter path to save segment config for {volume_name}", fg="cyan"
            ),
            type=click.Path(),
            default=os.path.join(os.getcwd(), f"seg_{volume_name}.toml"),
        )
        save_config(config, config_path)


@prepare.command("evaluate")
def prep_eval_config():
    """Create evaluation config files."""
    volumes = get_volumes()

    out_segs_prefix = click.prompt(
        click.style("Enter prefix for segmentation datasets", fg="cyan"),
        type=str,
    )
    pred_datasets = click.prompt(
        click.style("Enter prediction datasets (comma-separated)", fg="cyan"),
        type=str,
    )
    pred_datasets = [x.strip() for x in pred_datasets.split(",")]

    ret = create_evaluation_configs(volumes, out_segs_prefix, pred_datasets)

    for volume_name, config in ret["configs"].items():
        click.echo()
        config_path = click.prompt(
            click.style(f"Enter path to save eval config for {volume_name}", fg="cyan"),
            type=click.Path(),
            default=os.path.join(os.getcwd(), f"eval_{volume_name}.toml"),
        )
        save_config(config, config_path)


@prepare.command("filter")
def prep_filter_config():
    """Create config files for filtering segmentations."""
    volumes = get_volumes()

    out_segs_prefix = click.prompt(
        click.style("Enter prefix for segmentation datasets", fg="cyan"),
        type=str,
    )
    eval_dir = click.prompt(
        click.style("Enter path to evaluation directory", fg="cyan"),
        type=click.Path(),
    )

    ret = create_filter_configs(volumes, out_segs_prefix, eval_dir)

    for volume_name, config in ret["configs"].items():
        click.echo()
        config_path = click.prompt(
            click.style(
                f"Enter path to save filter config for {volume_name}", fg="cyan"
            ),
            type=click.Path(),
            default=os.path.join(os.getcwd(), f"filter_{volume_name}.toml"),
        )
        save_config(config, config_path)
