import click
import os
from glob import glob
from shutil import copytree, ignore_patterns
import requests
import zipfile


from .styles import cli_echo, cli_confirm


BS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
MODEL_DIR = os.path.join(BS_DIR, "models")
EVAL_ERROR_THRESHOLDS = {"lsds": [0.01, 1.0], "affs": [0.1, 1.0]}

MODEL_NAMES = sorted(
        [
            d
            for d in os.listdir(MODEL_DIR)
            if os.path.isdir(os.path.join(MODEL_DIR, d))
        ]
    )
MODEL_URLS = {
    "3d_affs_from_2d_affs": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.3.2/3d_affs_from_2d_affs.zip",
    "3d_affs_from_2d_lsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.3.2/3d_affs_from_2d_lsd.zip",
    "3d_affs_from_2d_mtlsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.3.2/3d_affs_from_2d_mtlsd.zip",
    "3d_affs_from_3d_lsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.3.2/3d_affs_from_3d_lsd.zip",
}


def checkpoint_iteration(checkpoint):
    """Training iteration in a checkpoint name, or None if the name carries none.

    `model_checkpoint_5000` and `model_checkpoint_5000.ckpt` both give 5000."""
    name = os.path.splitext(os.path.basename(checkpoint))[0]
    iteration = name.rsplit("_", 1)[-1]
    return int(iteration) if iteration.isdigit() else None


def output_level(checkpoint, chain_str=""):
    """The group predict writes its outputs in: <iteration>, or <iteration>--from--<chain_str>."""
    iteration = checkpoint_iteration(checkpoint)
    return f"{iteration}--from--{chain_str}" if chain_str else str(iteration)


def find_checkpoints(setup_dir):
    """Checkpoints of a setup by iteration. Look one level down as well."""
    paths = glob(os.path.join(setup_dir, "*", "model_checkpoint_*")) + glob(
        os.path.join(setup_dir, "model_checkpoint_*")
    )
    iterations = {checkpoint_iteration(p): p for p in paths}
    iterations.pop(None, None)
    return iterations


def copy_model_scripts(model_name, setup_dir, style="train", cli_edit=True):
    src = os.path.abspath(os.path.join(BS_DIR, "models", model_name))
    copytree(src, setup_dir, dirs_exist_ok=True, ignore=ignore_patterns("model_checkpoint_*", "__pycache__"))
    rel = os.path.relpath(setup_dir)
    cli_echo(f"Copied the {model_name} scripts into {setup_dir if rel.startswith('..') else rel}", style)

    # edit net config ?
    net_config_path = os.path.join(setup_dir, "net_config.json")
    if cli_edit:
        if cli_confirm(f"Edit {net_config_path}?", style, default=False):
            click.edit(filename=net_config_path)


def download_checkpoints(model_name, setup_dir, style="prepare"):
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}")

    url = MODEL_URLS[model_name]
    file_path = os.path.join(setup_dir, "checkpoints.zip")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(file_path, "wb") as file:
            for data in response.iter_content(chunk_size=1 << 20):
                file.write(data)
    except requests.RequestException as e:
        raise click.ClickException(f"download of {model_name} checkpoints from {url} failed: {e}")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(setup_dir)
        checkpoints = [n for n in zip_ref.namelist() if os.path.basename(n).startswith("model_checkpoint_")]
    os.remove(file_path)

    if not checkpoints:
        raise click.ClickException(f"{url} holds no model_checkpoint_* file for {setup_dir}")
    for name in checkpoints:
        size = os.path.getsize(os.path.join(setup_dir, name)) / 1e6
        cli_echo(f"Downloaded the {model_name} checkpoint ({size:.0f} MB)", style)


def checkpoint_path(setup_dir, checkpoint):
    """The file of a checkpoint: the path, the path with .ckpt, or the setup's checkpoint of
    the same iteration, one level down as well. None when there is none."""
    for path in (checkpoint, checkpoint + ".ckpt"):
        if os.path.exists(path):
            return path
    return find_checkpoints(setup_dir).get(checkpoint_iteration(checkpoint))
