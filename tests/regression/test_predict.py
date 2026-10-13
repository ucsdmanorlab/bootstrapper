"""Prediction covers the configured region, reads its checkpoint by iteration, uses only the
granted GPUs, and refuses an output it cannot write into."""

import importlib
import json
import os

import click
import numpy as np
import pytest
from funlib.persistence import prepare_ds

from bootstrapper.configs import checkpoint_iteration
from bootstrapper.predict import (
    call_predict,
    get_devices,
    get_pred_config,
    refuse_incompatible_output,
    resolve_checkpoint,
)

predict_module = importlib.import_module("bootstrapper.predict")

NET = {
    "inputs": ["raw"],
    "input_shape": [12, 40, 40],
    "output_shape": [8, 32, 32],
    "shape_increase": [0, 0, 0],
    "outputs": {"affs": {"dims": 3, "dtype": "uint8"}},
}


@pytest.fixture(autouse=True)
def one_granted_gpu(monkeypatch):
    """get_pred_config reads the grant, and the suite may run with an empty one."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


def make_config(tmp_path, **keys):
    """The keys of a [predict] step: a fake setup and a 30 x 64 x 64 raw array."""
    setup_dir = tmp_path / "setup"
    setup_dir.mkdir()
    (setup_dir / "net_config.json").write_text(json.dumps(NET))
    (setup_dir / "model_checkpoint_10").touch()
    (setup_dir / "predict.py").write_text("")
    raw = str(tmp_path / "data.zarr" / "raw")
    prepare_ds(raw, shape=(30, 64, 64), voxel_size=(1, 1, 1), dtype=np.uint8)
    return {
        "setup_dir": str(setup_dir),
        "checkpoint": str(setup_dir / "model_checkpoint_10"),
        "input_datasets": [raw],
        "output_datasets_prefix": str(tmp_path / "out.zarr"),
        "num_gpus": 1,
        **keys,
    }


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

    monkeypatch.setattr(
        predict_module.subprocess,
        "run",
        lambda cmd, check, env: seen.append(env["CUDA_VISIBLE_DEVICES"]),
    )
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
