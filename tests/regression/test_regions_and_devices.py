"""Prediction must cover the volume's own region and use only the granted GPUs."""

import importlib
import json
import os

import numpy as np
import pytest
import toml
from funlib.persistence import prepare_ds

from bootstrapper.configs import (
    checkpoint_iteration,
    create_prediction_configs,
    find_checkpoints,
    get_sub_roi,
)
from bootstrapper.predict import call_predict, get_devices, get_pred_config

predict_module = importlib.import_module("bootstrapper.predict")

NET_CONFIG = {
    "inputs": {"raw": {"dims": 1}},
    "outputs": {"3d_affs": {"dims": 3, "dtype": "uint8"}},
    "input_shape": [24, 148, 148],
    "output_shape": [4, 56, 56],
    "shape_increase": [0, 0, 0],
}


def write_array(tmp_path, offset=(4000, 2000, 2000), name="raw"):
    path = str(tmp_path / "vol.zarr" / name)
    prepare_ds(
        path,
        shape=(100, 200, 200),
        offset=offset,
        voxel_size=(50, 8, 8),
        axis_names=["z", "y", "x"],
        dtype=np.uint8,
    )
    return path


def write_setup(tmp_path, name="setup_01", checkpoint="model_checkpoint_5000.ckpt"):
    setup_dir = tmp_path / name
    setup_dir.mkdir(parents=True)
    (setup_dir / "net_config.json").write_text(json.dumps(NET_CONFIG))
    (setup_dir / "predict.py").write_text("")
    (setup_dir / checkpoint).write_text("")
    return str(setup_dir)


def write_pred_config(tmp_path, setup_dir, raw, checkpoint, **extra):
    config_file = str(tmp_path / "02_pred.toml")
    with open(config_file, "w") as f:
        toml.dump(
            {
                "01-setup_01": {
                    "setup_dir": setup_dir,
                    "input_datasets": [raw],
                    "output_datasets_prefix": str(tmp_path / "vol.zarr" / "setup_01"),
                    "checkpoint": checkpoint,
                    "chain_str": "",
                    "num_workers": 1,
                    "num_gpus": 1,
                    **extra,
                }
            },
            f,
        )
    return config_file


# --- the region a cropped volume predicts over ---


def test_sub_roi_of_a_cropped_volume_is_its_own_content(tmp_path):
    raw = write_array(tmp_path, offset=(4000, 2000, 2000))

    offset, shape, voxel_size = get_sub_roi(raw, offset=(0, 0, 0), shape=(100, 200, 200))

    assert (offset, shape) == ((4000, 2000, 2000), (5000, 1600, 1600))
    assert voxel_size == (50, 8, 8)


def test_sub_roi_defaults_take_the_whole_cropped_volume(tmp_path, monkeypatch):
    monkeypatch.setattr("bootstrapper.configs.cli_prompt", lambda *a, **kw: "0 0 0")
    raw = write_array(tmp_path, offset=(4000, 2000, 2000))

    offset, shape, _ = get_sub_roi(raw)

    assert (offset, shape) == ((4000, 2000, 2000), (5000, 1600, 1600))


def test_sub_roi_offset_counts_from_the_start_of_the_array(tmp_path):
    raw = write_array(tmp_path, offset=(4000, 2000, 2000))

    offset, shape, _ = get_sub_roi(raw, offset=(10, 10, 10), shape=(20, 20, 20))

    assert (offset, shape) == ((4500, 2080, 2080), (1000, 160, 160))


def test_sub_roi_clips_a_shape_that_runs_past_the_end(tmp_path):
    raw = write_array(tmp_path, offset=(4000, 2000, 2000))

    _, shape, _ = get_sub_roi(raw, offset=(10, 10, 10), shape=(999, 999, 999))

    assert shape == (4500, 1520, 1520)


def test_sub_roi_of_a_volume_at_the_origin_is_unchanged(tmp_path):
    raw = write_array(tmp_path, offset=(0, 0, 0))

    assert get_sub_roi(raw, offset=(0, 0, 0), shape=(100, 200, 200))[:2] == (
        (0, 0, 0),
        (5000, 1600, 1600),
    )
    assert get_sub_roi(raw, offset=(10, 10, 10), shape=(20, 20, 20))[:2] == (
        (500, 80, 80),
        (1000, 160, 160),
    )


@pytest.mark.parametrize(
    "roi_offset, roi_shape",
    [
        ("4200 2100 2100", "1000 1000 1000"),
        ("4200,2100,2100", "1000,1000,1000"),
        ([4200, 2100, 2100], [1000, 1000, 1000]),
    ],
)
def test_region_options_accept_strings(tmp_path, roi_offset, roi_shape):
    setup_dir = write_setup(tmp_path)
    raw = write_array(tmp_path)
    config_file = write_pred_config(
        tmp_path, setup_dir, raw, os.path.join(setup_dir, "model_checkpoint_5000.ckpt")
    )

    config = get_pred_config(
        config_file, "01-setup_01", roi_offset=roi_offset, roi_shape=roi_shape
    )

    args = config["args"]
    assert args[args.index("-ro") + 1] == "4200 2100 2100"
    assert args[args.index("-rs") + 1] == "1000 1000 1000"


# --- checkpoints ---


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model_checkpoint_5000", 5000),
        ("model_checkpoint_5000.ckpt", 5000),
        ("/a/b/model_checkpoint_25000.ckpt", 25000),
        ("last.ckpt", None),
        ("model_checkpoint_5000-v1.ckpt", None),
    ],
)
def test_checkpoint_iteration(name, expected):
    assert checkpoint_iteration(name) == expected


def test_find_checkpoints_looks_inside_a_nested_setup_directory(tmp_path):
    setup_dir = tmp_path / "3d_affs_from_2d_lsd"
    nested = setup_dir / "3d_affs_from_2d_lsd"
    nested.mkdir(parents=True)
    (nested / "model_checkpoint_25000.ckpt").write_text("")
    (setup_dir / "last.ckpt").write_text("")

    assert find_checkpoints(str(setup_dir)) == {
        25000: str(nested / "model_checkpoint_25000.ckpt")
    }


def test_output_dataset_is_named_by_the_iteration_alone(tmp_path):
    setup_dir = write_setup(tmp_path)
    raw = write_array(tmp_path)
    config_file = write_pred_config(
        tmp_path, setup_dir, raw, os.path.join(setup_dir, "model_checkpoint_5000.ckpt")
    )

    config = get_pred_config(config_file, "01-setup_01")

    args = config["args"]
    assert args[args.index("-o") + 1].endswith("/setup_01/5000/3d_affs")


def test_an_unnamed_checkpoint_is_refused_before_any_output_is_made(tmp_path):
    setup_dir = write_setup(tmp_path, checkpoint="last.ckpt")
    raw = write_array(tmp_path)
    config_file = write_pred_config(
        tmp_path, setup_dir, raw, os.path.join(setup_dir, "last.ckpt")
    )

    with pytest.raises(ValueError, match="last.ckpt"):
        get_pred_config(config_file, "01-setup_01")

    assert not (tmp_path / "vol.zarr" / "setup_01").exists()


def test_default_iteration_stops_at_the_iteration_trained(tmp_path, monkeypatch):
    setup_dir = write_setup(tmp_path)
    (tmp_path / "setup_01" / "3d_affs_from_2d_lsd").mkdir()
    nested = tmp_path / "setup_01" / "3d_affs_from_2d_lsd" / "model_checkpoint_2000.ckpt"
    nested.write_text("")
    raw = write_array(tmp_path)

    defaults = {}

    def fake_prompt(text, style=None, **kwargs):
        if "checkpoint iteration" in text:
            defaults[text] = kwargs["default"]
            return kwargs["default"]
        return kwargs.get("default", "0 0 0")

    monkeypatch.setattr("bootstrapper.configs.cli_prompt", fake_prompt)
    monkeypatch.setattr("bootstrapper.configs.cli_confirm", lambda *a, **kw: False)

    volumes = {
        "vol": {
            "output_container": str(tmp_path / "vol.zarr"),
            "raw_dataset": raw,
            "voxel_size": [50, 8, 8],
        }
    }
    out = create_prediction_configs(volumes, [setup_dir])

    # 5000 is on disk, 2000 is nested; the default may not exceed the highest trained
    assert list(defaults.values()) == [5000]
    assert out["configs"]["vol"]["01-setup_01"]["checkpoint"] == os.path.join(
        setup_dir, "model_checkpoint_5000.ckpt"
    )


# --- the GPUs a prediction may use ---


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

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        get_devices(2)


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
    setup_dir = write_setup(tmp_path)
    raw = write_array(tmp_path)
    config_file = write_pred_config(
        tmp_path,
        setup_dir,
        raw,
        os.path.join(setup_dir, "model_checkpoint_5000.ckpt"),
        num_gpus=4,
    )

    config = get_pred_config(config_file, "01-setup_01")

    assert config["num_gpus"] == 1
    assert "-d" not in config["args"]
