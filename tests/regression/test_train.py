"""Training must use only the granted GPUs and say so when it runs on the CPU; -s is the step,
and a model script names both dirs when the setup_dir is not its own."""

import ast
import importlib
import logging
import os
import runpy
import sys

import pytest
import toml
import zarr
from click.testing import CliRunner

import bootstrapper.training as training
from bootstrapper.configs import MODEL_DIR, MODEL_NAMES
from bootstrapper.train import setup_train

# bootstrapper/__init__.py rebinds the name "train" to the command
train_module = importlib.import_module("bootstrapper.train")


def write_config(tmp_path, **extra):
    """A one-step train file, [train.setup_01], with one sample."""
    setup_dir = tmp_path / "setup_01"
    setup_dir.mkdir()
    container = tmp_path / "data.zarr"
    for array in ("raw", "labels"):
        zarr.open(str(container / array), mode="w", shape=(4, 4, 4), dtype="uint8")[
            :
        ] = 1

    config = {
        "setup_dir": str(setup_dir),
        "voxel_size": [50, 8, 8],
        "max_iterations": 10,
        "save_checkpoints_every": 5,
        "save_snapshots_every": 5,
        "samples": [
            {
                "raw": str(container / "raw"),
                "labels": str(container / "labels"),
            }
        ],
    }
    config.update(extra)
    config_file = tmp_path / "01_train.toml"
    config_file.write_text(toml.dumps({"train": {"setup_01": config}}))
    return str(config_file)


class FakeTrainer:
    """Records what fit() asks the Trainer for, without training."""

    kwargs = {}

    def __init__(self, **kwargs):
        FakeTrainer.kwargs = kwargs

    def fit(self, *args, **kwargs):
        pass


class EmptyDataset(training.IterableDataset):
    def __iter__(self):
        return iter([])


@pytest.mark.parametrize("devices,expected", [(None, "auto"), ([0], [0])])
def test_fit_passes_devices_to_trainer(tmp_path, monkeypatch, devices, expected):
    monkeypatch.setattr(training.pl, "Trainer", FakeTrainer)

    training.fit(
        object(),
        EmptyDataset(),
        str(tmp_path),
        max_iterations=10,
        save_checkpoints_every=5,
        snapshot_callback=object(),
        num_workers=0,
        devices=devices,
    )

    assert FakeTrainer.kwargs["devices"] == expected


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_script_accepts_devices(model_name):
    script = os.path.join(MODEL_DIR, model_name, "train.py")
    tree = ast.parse(open(script).read(), script)
    train_fn = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"
    ][0]

    assert "devices" in [a.arg for a in train_fn.args.args]


def test_devices_reach_the_worker_config(tmp_path):
    config_file = write_config(tmp_path, devices=[0])

    train_script, worker_config = setup_train(config_file)

    assert train_script == str(tmp_path / "setup_01" / "train.py")
    assert worker_config == str(tmp_path / "setup_01" / "train.toml")
    assert toml.load(worker_config)["devices"] == [0]


def test_overrides_reach_the_worker_config_and_leave_the_file(tmp_path):
    config_file = write_config(tmp_path)

    _, worker_config = setup_train(config_file, max_iterations=99)

    assert toml.load(worker_config)["max_iterations"] == 99
    assert toml.load(config_file)["train"]["setup_01"]["max_iterations"] == 10


def test_missing_labels_is_refused_by_path(tmp_path):
    config_file = write_config(tmp_path)
    config = toml.load(config_file)
    absent = str(tmp_path / "data.zarr" / "absent")
    config["train"]["setup_01"]["samples"][0]["labels"] = absent
    open(config_file, "w").write(toml.dumps(config))

    with pytest.raises(ValueError, match="labels") as error:
        setup_train(config_file)

    assert absent in str(error.value)


@pytest.mark.parametrize(
    "grant,why",
    [
        ("", "CUDA_VISIBLE_DEVICES is set but empty"),
        (None, "no CUDA device is visible"),
    ],
)
def test_cpu_training_warns_with_the_reason(monkeypatch, grant, why):
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    if grant is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", grant)

    assert why in training.device_warning()


def test_train_s_is_the_step(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        train_module, "run_training", lambda config, step, **kw: seen.update(step=step)
    )
    config = tmp_path / "train.toml"
    config.write_text("")

    result = CliRunner().invoke(train_module.train, [str(config), "-s", "apical"])

    assert result.exit_code == 0, result.output
    assert seen["step"] == "apical"


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_a_model_script_names_both_dirs_when_setup_dir_differs(
    model_name, tmp_path, monkeypatch
):
    script = os.path.join(MODEL_DIR, model_name, "train.py")
    setup_dir = os.path.realpath(os.path.dirname(script))
    config = tmp_path / "train.toml"
    config.write_text('setup_dir = "/elsewhere"\n')
    monkeypatch.setattr(sys, "argv", ["train.py", str(config)])
    monkeypatch.syspath_prepend(setup_dir)
    monkeypatch.delitem(sys.modules, "model", raising=False)
    monkeypatch.setattr(logging.getLogger(), "level", logging.getLogger().level)

    with pytest.raises(SystemExit) as stop:
        runpy.run_path(script, run_name="__main__")

    assert "/elsewhere" in str(stop.value) and setup_dir in str(stop.value)
