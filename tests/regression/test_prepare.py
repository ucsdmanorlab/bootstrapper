"""bs prepare asks about the data, then writes volumes.toml, numbered step files and a
checkpoint-free setup with no __pycache__; every file loads."""

import glob
import os

import numpy as np
import pytest
import toml
from click.testing import CliRunner
from funlib.persistence import prepare_ds

from bootstrapper import configs
from bootstrapper.config import load
from bootstrapper.prepare import prepare

FILES = [
    "01_train_2d_mtlsd.toml",
    "02_predict_2d_mtlsd.toml",
    "03_predict_3d_affs_from_2d_mtlsd.toml",
    "04_segment.toml",
    "05_refine.toml",
    "06_evaluate.toml",
]


def run_prepare(tmp_path, models=""):
    """One volume, sparse labels on two sections, background beside each object."""
    labels = np.zeros((8, 64, 64), np.uint64)
    labels[2, 10:30, 10:30] = 1
    labels[5, 30:50, 30:50] = 2
    for name, data in [
        ("raw", np.full(labels.shape, 100, np.uint8)),
        ("labels", labels),
    ]:
        ds = prepare_ds(
            f"{tmp_path}/apical.zarr/{name}",
            shape=labels.shape,
            offset=(0, 0, 0),
            voxel_size=(50, 8, 8),
            axis_names=["z", "y", "x"],
            units=["nm", "nm", "nm"],
            dtype=data.dtype,
        )
        ds[ds.roi] = data
    answers = [
        f"{tmp_path}/apical.zarr/raw",
        f"{tmp_path}/apical.zarr/labels",
        "n",  # not every object is labelled
        "",  # labels on single sections: the default, guessed from the labels
        "n",  # untraced space next to an object is background
        "",  # no mask
        f"{tmp_path}/apical.zarr/labels",  # ground truth
        "n",  # no other volume
        models,  # "": the default chain for 2D labels
        "",  # GPUs
    ]

    return CliRunner().invoke(
        prepare,
        ["-o", f"{tmp_path}/round_01"],
        input="\n".join(answers) + "\n",
        catch_exceptions=False,
    )


@pytest.fixture
def dialogue(tmp_path):
    return run_prepare(tmp_path)


def step(tmp_path, name):
    return load(f"{tmp_path}/round_01/{name}")[0]


def test_the_dialogue_writes_the_files_in_run_order(tmp_path, dialogue):
    assert dialogue.exit_code == 0, dialogue.output
    files = sorted(
        os.path.basename(p) for p in glob.glob(f"{tmp_path}/round_01/*.toml")
    )
    assert files == FILES + ["volumes.toml"]
    assert dialogue.output.count("wrote ") == 7
    assert "bs run" in dialogue.output
    for name in FILES:
        assert load(f"{tmp_path}/round_01/{name}")


def test_the_volume_facts_are_recorded(tmp_path, dialogue):
    volume = toml.load(f"{tmp_path}/round_01/volumes.toml")["volumes"]["apical"]

    assert volume == {
        "raw": f"{tmp_path}/apical.zarr/raw",
        "labels": f"{tmp_path}/apical.zarr/labels",
        "training_mask": "labels",
        "2d_labels": True,
        "mask_dilation": 10,
        "gt": f"{tmp_path}/apical.zarr/labels",
    }


def test_the_train_step_trains_a_fresh_setup_on_a_labels_mask(tmp_path, dialogue):
    train = step(tmp_path, "01_train_2d_mtlsd.toml")

    assert train.keys["setup_dir"] == f"{tmp_path}/round_01/2d_mtlsd"
    assert (
        train.keys["samples"][0]["mask"] == f"{tmp_path}/apical.zarr/masks/labels_mask"
    )
    assert os.path.isdir(f"{tmp_path}/apical.zarr/masks/labels_mask")
    assert os.path.exists(f"{tmp_path}/round_01/2d_mtlsd/train.py")
    assert not glob.glob(f"{tmp_path}/round_01/2d_mtlsd/model_checkpoint_*")


def test_the_second_predict_step_reads_the_first_ones_output(tmp_path, dialogue):
    predict = step(tmp_path, "03_predict_3d_affs_from_2d_mtlsd.toml")

    assert predict.keys["setup_dir"].endswith(
        "bootstrapper/models/3d_affs_from_2d_mtlsd"
    )
    assert predict.volumes["apical"]["chain_str"] == "2d_mtlsd_30000"
    assert predict.volumes["apical"]["input_datasets"][0].endswith(
        "2d_mtlsd/30000/2d_lsds"
    )


def test_a_model_after_2d_mtlsd_reads_only_the_outputs_it_takes(tmp_path):
    result = run_prepare(tmp_path, "2M 3Af2L")

    assert result.exit_code == 0, result.output
    predict = step(tmp_path, "03_predict_3d_affs_from_2d_lsd.toml")
    assert predict.volumes["apical"]["input_datasets"] == [
        f"{tmp_path}/apical.zarr/2d_mtlsd/30000/2d_lsds"
    ]


def test_segment_refine_and_evaluate_follow_the_affs(tmp_path, dialogue):
    segment = step(tmp_path, "04_segment.toml")
    refine = step(tmp_path, "05_refine.toml")
    evaluate = step(tmp_path, "06_evaluate.toml")

    params = segment.keys["params"]
    assert segment.keys["method"] == "mws"
    assert len(params["aff_neighborhood"]) == len(params["bias"])
    assert segment.volumes["apical"]["affs_dataset"].endswith(
        "--from--2d_mtlsd_30000/3d_affs"
    )
    assert list(refine.keys["ops"]) == ["fill-holes"]
    assert refine.volumes["apical"]["in_array"].startswith(
        segment.volumes["apical"]["seg_dataset_prefix"] + "/gb1_-0.5"
    )
    assert (
        evaluate.volumes["apical"]["gt"]["labels_dataset"]
        == f"{tmp_path}/apical.zarr/labels"
    )
    assert evaluate.volumes["apical"]["pred"]["thresholds"] == [0.1, 1.0]


def test_model_copy_leaves_out_pycache(tmp_path, monkeypatch):
    src = tmp_path / "bs" / "models" / "m"
    (src / "__pycache__").mkdir(parents=True)
    (src / "train.py").write_text("")
    (src / "__pycache__" / "train.cpython-312.pyc").write_text("")
    monkeypatch.setattr(configs, "BS_DIR", str(tmp_path / "bs"))

    configs.copy_model_scripts("m", str(tmp_path / "setup"), cli_edit=False)

    assert os.listdir(tmp_path / "setup") == ["train.py"]
