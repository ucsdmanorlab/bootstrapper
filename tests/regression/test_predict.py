"""Prediction covers the configured region, reads its checkpoint by iteration, uses only the
granted GPUs, and refuses an output it cannot write into. Several workers run daisy blocks on
the zarr chunk grid, and a stopped worker takes its model process with it."""

import importlib
import json
import os
import subprocess
import sys
import time

import click
import numpy as np
import pytest
import toml
import zarr
from funlib.persistence import prepare_ds

from bootstrapper.configs import checkpoint_iteration
from bootstrapper.predict import (
    call_predict,
    get_devices,
    get_pred_config,
    refuse_incompatible_output,
    resolve_checkpoint,
    run_prediction,
)

predict_module = importlib.import_module("bootstrapper.predict")

NET = {
    "inputs": ["raw"],
    "input_shape": [12, 40, 40],
    "output_shape": [8, 32, 32],
    "shape_increase": [0, 0, 0],
    "outputs": {"affs": {"dims": 3, "dtype": "uint8"}},
}

WORKER_ARGS = """
import argparse
import sys

parser = argparse.ArgumentParser()
for flag in ("-c", "-n", "-ro", "-rs"):
    parser.add_argument(flag)
parser.add_argument("-i", action="append")
parser.add_argument("-o", action="append")
parser.add_argument("-d", action="store_true")
args = parser.parse_args()
"""

CONSTANT_WORKER = (
    WORKER_ARGS
    + """
import gunpowder as gp
import numpy as np

if not args.d:
    sys.exit("this fake worker runs only as a daisy worker (-d)")

affs = gp.ArrayKey("AFFS")
out = args.o[0]


class Constant(gp.BatchProvider):
    def setup(self):
        self.provides(affs, gp.ArraySpec(voxel_size=gp.Coordinate(1, 1, 1), dtype=np.uint8, interpolatable=False))

    def provide(self, request):
        spec = self.spec[affs].copy()
        spec.roi = request[affs].roi
        batch = gp.Batch()
        batch[affs] = gp.Array(np.full((3, *spec.roi.shape), 200, np.uint8), spec)
        return batch


reference = gp.BatchRequest()
reference.add(affs, gp.Coordinate(8, 32, 32))
pipeline = (
    Constant()
    + gp.ZarrWrite({affs: out.split(".zarr")[-1]}, store=out.split(".zarr")[0] + ".zarr")
    + gp.DaisyRequestBlocks(reference, roi_map={affs: "write_roi"})
)
with gp.build(pipeline):
    pipeline.request_batch(gp.BatchRequest())
"""
)

# the first worker to start waits for the second to sleep, then fails
ONE_FAILS_WORKER = """
import os
import sys
import time
from pathlib import Path

here = Path(__file__).parent
try:
    os.close(os.open(here / "first", os.O_CREAT | os.O_EXCL))
except FileExistsError:
    (here / "sleeper.pid").write_text(str(os.getpid()))
    time.sleep(120)
for _ in range(200):
    if (here / "sleeper.pid").exists():
        break
    time.sleep(0.1)
sys.exit(1)
"""


@pytest.fixture(autouse=True)
def one_granted_gpu(monkeypatch):
    """get_pred_config reads the grant, and the suite may run with an empty one."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


def make_config(tmp_path, worker="", z=30, **keys):
    """The keys of a [predict] step: a fake setup and a z x 64 x 64 raw array."""
    setup_dir = tmp_path / "setup"
    setup_dir.mkdir()
    (setup_dir / "net_config.json").write_text(json.dumps(NET))
    (setup_dir / "model_checkpoint_10").touch()
    (setup_dir / "predict.py").write_text(worker)
    raw = str(tmp_path / "data.zarr" / "raw")
    prepare_ds(raw, shape=(z, 64, 64), voxel_size=(1, 1, 1), dtype=np.uint8)
    return {
        "setup_dir": str(setup_dir),
        "checkpoint": str(setup_dir / "model_checkpoint_10"),
        "input_datasets": [raw],
        "output_datasets_prefix": str(tmp_path / "out.zarr"),
        "num_gpus": 1,
        **keys,
    }


def run_bs_predict(tmp_path, config):
    config_file = tmp_path / "predict.toml"
    config_file.write_text(toml.dumps({"predict": config}))
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from bootstrapper.cli import cli; cli()",
            "predict",
            str(config_file),
        ],
        cwd=tmp_path,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "-1"},
        capture_output=True,
        text=True,
        timeout=60,
    )


def alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize(
    "roi_offset, roi_shape",
    [
        ("8 0 0", "16 64 64"),
        ("8,0,0", "16,64,64"),
        ([8, 0, 0], [16, 64, 64]),
    ],
)
def test_region_options_accept_strings(tmp_path, roi_offset, roi_shape):
    config = get_pred_config(
        make_config(tmp_path), roi_offset=roi_offset, roi_shape=roi_shape
    )

    args = config["args"]
    assert args[args.index("-ro") + 1] == "8 0 0"
    assert args[args.index("-rs") + 1] == "16 64 64"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model_checkpoint_5000", 5000),
        ("/a/b/model_checkpoint_25000.ckpt", 25000),
        ("model_checkpoint_5000-v1.ckpt", None),
    ],
)
def test_checkpoint_iteration(name, expected):
    assert checkpoint_iteration(name) == expected


def test_output_dataset_is_named_by_the_iteration_alone(tmp_path):
    args = get_pred_config(make_config(tmp_path))["args"]

    assert args[args.index("-o") + 1] == str(tmp_path / "out.zarr" / "10" / "affs")


def test_an_unnamed_checkpoint_is_refused_before_any_output_is_made(tmp_path):
    config = make_config(tmp_path)
    (tmp_path / "setup" / "last.ckpt").touch()
    config["checkpoint"] = str(tmp_path / "setup" / "last.ckpt")

    with pytest.raises(ValueError, match="last.ckpt"):
        get_pred_config(config)

    assert not (tmp_path / "out.zarr").exists()


def test_a_nested_checkpoint_resolves_from_the_setup_path(tmp_path):
    (tmp_path / "nested").mkdir()
    nested = tmp_path / "nested" / "model_checkpoint_25000.ckpt"
    nested.write_text("")

    assert resolve_checkpoint(
        str(tmp_path), f"{tmp_path}/model_checkpoint_25000"
    ) == str(nested)
    assert resolve_checkpoint(str(tmp_path), str(nested)) == str(nested)


def test_a_missing_iteration_is_refused_with_the_iterations_there(tmp_path):
    (tmp_path / "model_checkpoint_25000.ckpt").write_text("")

    with pytest.raises(click.ClickException, match=r"30000.*\[25000\]"):
        resolve_checkpoint(str(tmp_path), f"{tmp_path}/model_checkpoint_30000")


def test_unset_devices_means_every_local_device(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert get_devices(4) == ["0", "1", "2", "3"]


def test_granted_devices_are_passed_through_verbatim(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1a2b, MIG-3c4d/1/0")

    assert get_devices(2) == ["GPU-1a2b", "MIG-3c4d/1/0"]


def test_more_gpus_requested_than_granted_are_clamped(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

    assert get_devices(4) == ["4", "5"]


def test_an_empty_device_list_is_refused(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES") as error:
        get_devices(2)

    assert "granted" in str(error.value) and "Unset" not in str(error.value)


def test_workers_index_into_the_granted_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    seen = []

    class Child:
        pid = None

        def __init__(self, cmd, env, start_new_session):
            seen.append(env["CUDA_VISIBLE_DEVICES"])

        def wait(self):
            return 0

        poll = wait

    monkeypatch.setattr(predict_module.subprocess, "Popen", Child)
    monkeypatch.setattr(predict_module.signal, "signal", lambda *args: None)
    config = {"devices": get_devices(2), "worker": "predict.py", "args": []}

    for worker_id in range(4):
        monkeypatch.setattr(
            predict_module.daisy.Context,
            "from_env",
            staticmethod(lambda worker_id=worker_id: {"worker_id": str(worker_id)}),
        )
        call_predict(config)

    assert seen == ["4", "5", "4", "5"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5"


def test_fewer_visible_devices_drop_prediction_off_the_blockwise_path(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    config = get_pred_config(make_config(tmp_path, num_gpus=4))

    assert config["num_gpus"] == 1
    assert "-d" not in config["args"]


@pytest.mark.parametrize(
    "grant,num_gpus,expected",
    [("4,5", 1, "4"), (None, 1, "0"), ("4,5", 0, "")],
)
def test_the_single_worker_gets_the_grant(
    tmp_path, monkeypatch, capsys, grant, num_gpus, expected
):
    if grant is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", grant)
    seen = []
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kwargs: seen.append(kwargs.get("env") or {})
    )
    config_file = tmp_path / "predict.toml"
    config_file.write_text(
        toml.dumps({"predict": make_config(tmp_path, num_gpus=num_gpus)})
    )

    run_prediction(str(config_file))

    assert seen[0].get("CUDA_VISIBLE_DEVICES") == expected
    warned = "CUDA_VISIBLE_DEVICES is not set" in capsys.readouterr().out
    assert warned == (grant is None)


def test_a_roi_smaller_than_one_block_names_both_shapes(tmp_path):
    config = make_config(tmp_path, roi_offset="0 0 0", roi_shape="4 64 64")

    with pytest.raises(click.ClickException) as error:
        get_pred_config(config)

    assert "(4, 64, 64)" in error.value.message
    assert "(8, 32, 32)" in error.value.message


def test_two_workers_write_every_voxel(tmp_path):
    result = run_bs_predict(
        tmp_path, make_config(tmp_path, CONSTANT_WORKER, num_workers=2)
    )

    assert result.returncode == 0, result.stdout + result.stderr
    affs = zarr.open(str(tmp_path / "out.zarr" / "10" / "affs"), "r")[:]
    assert affs.shape == (3, 30, 64, 64)
    assert (affs == 0).sum() == 0, (
        f"zero voxels at z {sorted(set(np.nonzero(affs == 0)[1].tolist()))}"
    )


def test_a_failed_worker_ends_the_run_and_stops_the_other_model_process(tmp_path):
    result = run_bs_predict(
        tmp_path, make_config(tmp_path, ONE_FAILS_WORKER, num_workers=2)
    )
    pid_file = tmp_path / "setup" / "sleeper.pid"
    pid = int(pid_file.read_text()) if pid_file.exists() else None

    try:
        assert result.returncode != 0
        assert pid is not None, "only one model process started"
        for _ in range(50):
            if not alive(pid):
                break
            time.sleep(0.1)
        assert not alive(pid), "the model process of the stopped worker still runs"
    finally:
        if pid is not None and alive(pid):
            os.kill(pid, 9)


SAME = dict(
    shape=(3, 4, 8, 8),
    offset=(0, 0, 0),
    voxel_size=(50, 8, 8),
    units=["nm"] * 3,
    dtype="uint8",
)


@pytest.fixture
def output(tmp_path):
    """An existing 3-channel prediction output."""
    path = f"{tmp_path}/o.zarr/3d_affs"
    prepare_ds(
        path,
        shape=(3, 4, 8, 8),
        offset=(0, 0, 0),
        voxel_size=(50, 8, 8),
        axis_names=["c^", "z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=np.uint8,
    )
    return path


def test_a_matching_or_missing_output_is_accepted(tmp_path, output):
    refuse_incompatible_output(output, **SAME)
    refuse_incompatible_output(f"{tmp_path}/o.zarr/none", **SAME)


def test_an_incompatible_output_is_refused_by_name(output):
    changed = {"voxel_size": (50, 16, 16), "shape": (6, 4, 8, 8), "dtype": "float32"}

    with pytest.raises(click.ClickException) as error:
        refuse_incompatible_output(output, **(SAME | changed))

    for needle in ["voxel_size", "(50, 16, 16)", "shape", "dtype"]:
        assert needle in str(error.value)
