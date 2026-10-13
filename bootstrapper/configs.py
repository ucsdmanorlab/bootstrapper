import click
import os
from glob import glob
from shutil import copytree, ignore_patterns
import requests
from tqdm import tqdm
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
MODEL_SHORT_NAMES = {
    "3d_affs_from_2d_affs": "3Af2A",
    "3d_affs_from_2d_lsd": "3Af2L",
    "3d_affs_from_2d_mtlsd": "3Af2M",
    "3d_affs_from_3d_lsd": "3Af3L",
}
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
    cli_echo(f"Copying {src} to {setup_dir}..", style)
    copytree(src, setup_dir, dirs_exist_ok=True, ignore=ignore_patterns("model_checkpoint_*"))

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

    cli_echo(f"Downloading {model_name} checkpoints zip to {setup_dir}...", style)
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise click.ClickException(f"download of {model_name} checkpoints from {url} failed: {e}")
    total_size = int(response.headers.get("content-length", 0))

    with open(file_path, "wb") as file, tqdm(
        desc=model_name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

    # unzip checkpoints
    cli_echo(f"Unzipping {model_name} checkpoints in {setup_dir}...", style)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(setup_dir)

    # clean up
    os.remove(file_path)
    if not find_checkpoints(setup_dir):
        raise click.ClickException(f"{url} holds no model_checkpoint_* file for {setup_dir}")
