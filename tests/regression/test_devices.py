"""Training must use only the granted GPUs, and an old setup dir must say so."""

import ast
import os

import pytest
import toml
import zarr

import bootstrapper.training as training
from bootstrapper.configs import MODEL_DIR, MODEL_NAMES
from bootstrapper.train import check_train_script, setup_train

OLD_SCRIPT = '''
def train(
    setup_dir,
    voxel_size,
    max_iterations,
    samples,
    save_checkpoints_every,
    save_snapshots_every,
):
    pass
'''

NEW_SCRIPT = OLD_SCRIPT.replace(
    "    save_snapshots_every,\n", "    save_snapshots_every,\n    devices=None,\n"
)


def write_setup(tmp_path, name, script):
    setup_dir = tmp_path / name
    setup_dir.mkdir()
    (setup_dir / "train.py").write_text(script)
    return str(setup_dir)


def write_config(tmp_path, setup_dir, name="train.toml", **extra):
    container = tmp_path / "data.zarr"
    for array in ("raw", "labels"):
        zarr.open(str(container / array), mode="w", shape=(4, 4, 4), dtype="uint8")[
            :
        ] = 1

    config = {
        "setup_dir": setup_dir,
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
    config_file = tmp_path / name
    config_file.write_text(toml.dumps(config))
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


@pytest.mark.parametrize(
    "devices,expected",
    [(None, "auto"), ("auto", "auto"), (1, 1), ([0], [0]), ("0,1", "0,1")],
)
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


def test_fit_defaults_to_auto(tmp_path, monkeypatch):
    monkeypatch.setattr(training.pl, "Trainer", FakeTrainer)

    training.fit(
        object(),
        EmptyDataset(),
        str(tmp_path),
        max_iterations=10,
        save_checkpoints_every=5,
        snapshot_callback=object(),
        num_workers=0,
    )

    assert FakeTrainer.kwargs["devices"] == "auto"


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_script_accepts_devices(model_name):
    script = os.path.join(MODEL_DIR, model_name, "train.py")
    tree = ast.parse(open(script).read(), script)
    train_fn = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"
    ][0]

    assert "devices" in [a.arg for a in train_fn.args.args]


def test_old_setup_dir_names_itself_and_the_refresh_command(tmp_path):
    setup_dir = write_setup(tmp_path, "setup_old", OLD_SCRIPT)
    config_file = write_config(tmp_path, setup_dir, devices=[0])

    with pytest.raises(ValueError) as error:
        setup_train(config_file)

    assert setup_dir in str(error.value)
    assert "bs prepare model" in str(error.value)


def test_old_setup_dir_without_devices_still_runs(tmp_path):
    setup_dir = write_setup(tmp_path, "setup_old", OLD_SCRIPT)
    config_file = write_config(tmp_path, setup_dir)

    train_script, used_config = setup_train(config_file)

    assert train_script == os.path.join(setup_dir, "train.py")
    assert used_config == config_file


def test_current_setup_dir_accepts_devices(tmp_path):
    setup_dir = write_setup(tmp_path, "setup_new", NEW_SCRIPT)
    config = {"setup_dir": setup_dir, "devices": [0], "max_iterations": 10}

    check_train_script(os.path.join(setup_dir, "train.py"), config)


def test_modified_config_keeps_one_toml_suffix(tmp_path):
    setup_dir = write_setup(tmp_path, "setup_new", NEW_SCRIPT)
    config_file = write_config(tmp_path, setup_dir)

    _, modified = setup_train(config_file, max_iterations=99)

    assert os.path.basename(modified) == "train_modified_0.toml"
    assert toml.load(modified)["max_iterations"] == 99


def test_missing_labels_names_the_step_that_makes_it(tmp_path):
    setup_dir = write_setup(tmp_path, "setup_new", NEW_SCRIPT)
    config_file = write_config(tmp_path, setup_dir)
    config = toml.load(config_file)
    config["samples"][0]["labels"] = str(tmp_path / "data.zarr" / "absent")
    open(config_file, "w").write(toml.dumps(config))

    with pytest.raises(ValueError) as error:
        setup_train(config_file)

    assert "bbox" in str(error.value)
    assert "bs prepare" in str(error.value)
