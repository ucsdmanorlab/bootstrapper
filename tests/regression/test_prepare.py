"""bs prepare asks about the data, then writes volumes.toml, numbered step files, a
checkpoint-free setup with no __pycache__, and the training masks in the round, never in the
data container; every file loads, and every param in it says what it does."""

import glob
import os
import re

import numpy as np
import pytest
import toml
from click.testing import CliRunner
from funlib.persistence import open_ds, prepare_ds

from bootstrapper import configs
from bootstrapper.catalog import parse_chain
from bootstrapper.config import load
from bootstrapper.params import KEYS
from bootstrapper.prepare import prepare, write_round

FILES = [
    "01_train_2d_mtlsd.toml",
    "02_predict_2d_mtlsd.toml",
    "03_predict_3d_affs_from_2d_mtlsd.toml",
    "04_segment.toml",
    "05_refine.toml",
    "06_evaluate.toml",
]


def write(path, data, chunk_shape=None):
    ds = prepare_ds(
        str(path),
        shape=data.shape,
        offset=(0, 0, 0),
        voxel_size=(50, 8, 8),
        axis_names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=data.dtype,
        chunk_shape=chunk_shape,
    )
    ds[ds.roi] = data
    return str(path)


def run_prepare(tmp_path, models=""):
    """One volume, sparse labels on two sections, background beside each object."""
    labels = np.zeros((8, 64, 64), np.uint64)
    labels[2, 10:30, 10:30] = 1
    labels[5, 30:50, 30:50] = 2
    write(tmp_path / "apical.zarr" / "raw", np.full(labels.shape, 100, np.uint8))
    write(tmp_path / "apical.zarr" / "labels", labels)
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


def test_each_param_says_what_it_does_and_each_example_loads(tmp_path, dialogue):
    """The step files show every key of params.KEYS, and volumes.toml each one it sets, with
    its meaning; with the commented lines on, every file still loads."""
    kinds = ["volumes", "train", "predict", "predict", "segment", "refine", "evaluate"]
    for name, kind in zip(["volumes.toml", *FILES], kinds):
        path = tmp_path / "round_01" / name
        text = path.read_text()
        volume = toml.loads(text).get("volumes", {}).get("apical", {})
        for table, keys in KEYS.items():
            for k, key in keys.items():
                if table.split(".")[0] == kind and (kind != "volumes" or k in volume):
                    assert re.search(rf"(?m)^(# )?{k} = ", text), (name, k)
                    assert key.help in text, (name, k)
        path.write_text(re.sub(r"(?m)^# (?=\[[\w.-]+\]$|[\w-]+ = )", "", text))
        load(path)


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


def test_the_train_step_trains_a_fresh_setup_on_a_mask_in_the_round(tmp_path, dialogue):
    train = step(tmp_path, "01_train_2d_mtlsd.toml")

    assert train.keys["setup_dir"] == f"{tmp_path}/round_01/2d_mtlsd"
    mask = f"{tmp_path}/round_01/training_mask.zarr/apical"
    assert train.keys["samples"][0]["mask"] == mask
    assert os.path.isdir(mask)
    assert sorted(os.listdir(f"{tmp_path}/apical.zarr")) == ["labels", "raw"]
    assert os.path.exists(f"{tmp_path}/round_01/2d_mtlsd/train.py")
    assert not glob.glob(f"{tmp_path}/round_01/2d_mtlsd/model_checkpoint_*")


def test_each_labelling_trains_on_its_own_mask(tmp_path):
    """Sparse labels train on labels > 0, grown mask_dilation voxels (in xy with 2D labels);
    dense labels on ones, or on the labelled sections with 2D labels; labels dense within a
    mask on that mask. The data container stays as it was."""
    labels = np.zeros((8, 16, 16), np.uint64)
    labels[2, 8, 8], labels[5, 4, 4] = 1, 2
    fg = labels > 0
    sections, within = np.zeros_like(fg), np.zeros_like(fg)
    sections[[2, 5]] = True
    within[:, :8] = True
    data = {
        "raw": write(tmp_path / "data.zarr" / "raw", np.zeros(labels.shape, np.uint8)),
        "labels": write(tmp_path / "data.zarr" / "labels", labels, (2, 16, 16)),
    }
    own = write(tmp_path / "data.zarr" / "within", within.astype(np.uint8))

    def grown(*axes):
        return fg | np.any([np.roll(fg, s, a) for a in axes for s in (1, -1)], 0)

    cases = {
        "sparse": ({"training_mask": "labels"}, fg),
        "sparse-grown": (
            {"training_mask": "labels", "mask_dilation": 1},
            grown(0, 1, 2),
        ),
        "sparse-2d-grown": (
            {"training_mask": "labels", "mask_dilation": 1, "2d_labels": True},
            grown(1, 2),
        ),
        "dense": ({"training_mask": "everywhere"}, np.ones_like(fg)),
        "dense-2d": ({"training_mask": "everywhere", "2d_labels": True}, sections),
        "within": ({"training_mask": own}, within),
    }
    container = sorted(glob.glob(f"{tmp_path}/data.zarr/**", recursive=True))

    volumes = {name: data | keys for name, (keys, _) in cases.items()}
    (tmp_path / "round_01").mkdir()
    write_round(f"{tmp_path}/round_01", volumes, parse_chain("3M"))

    samples = step(tmp_path, "01_train_3d_mtlsd.toml").keys["samples"]
    for (name, (_, expected)), sample in zip(cases.items(), samples):
        # no mask: the trainer takes labels > 0
        mask = open_ds(sample["mask"]) if "mask" in sample else None
        trained_on = mask[mask.roi] if mask else fg
        assert np.array_equal(trained_on, expected), name
    assert sorted(glob.glob(f"{tmp_path}/data.zarr/**", recursive=True)) == container


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
    assert evaluate.keys["thresholds"] == [0.1, 1.0]


def test_model_copy_leaves_out_pycache(tmp_path, monkeypatch):
    src = tmp_path / "bs" / "models" / "m"
    (src / "__pycache__").mkdir(parents=True)
    (src / "train.py").write_text("")
    (src / "__pycache__" / "train.cpython-312.pyc").write_text("")
    monkeypatch.setattr(configs, "BS_DIR", str(tmp_path / "bs"))

    configs.copy_model_scripts("m", str(tmp_path / "setup"), cli_edit=False)

    assert os.listdir(tmp_path / "setup") == ["train.py"]
