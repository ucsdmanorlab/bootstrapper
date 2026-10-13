"""A run prints progress, warnings, errors and results, not the input it was given: predict and
segment print no config dict, evaluate no bare ROI, training no Lightning tip and no warning about
the setup dir it resumes from. Evaluate ends with one line per volume and segmentation."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import toml
from funlib.persistence import prepare_ds

import bootstrapper.post.watershed_mutex as watershed_mutex
from bootstrapper.predict import run_prediction
from bootstrapper.segment import run_segmentation

NET = {
    "inputs": ["raw"],
    "input_shape": [12, 40, 40],
    "output_shape": [8, 32, 32],
    "shape_increase": [0, 0, 0],
    "outputs": {"affs": {"dims": 3, "dtype": "uint8"}},
}


def write(path, data, voxel_size=(50, 4, 4)):
    ds = prepare_ds(str(path), shape=data.shape, voxel_size=voxel_size, dtype=data.dtype, chunk_shape=data.shape[-3:])
    ds[ds.roi] = data
    return str(path)


def test_predict_prints_no_config_dict(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    setup = tmp_path / "setup"
    setup.mkdir()
    (setup / "net_config.json").write_text(json.dumps(NET))
    (setup / "model_checkpoint_10").touch()
    raw = write(tmp_path / "data.zarr" / "raw", np.zeros((30, 64, 64), np.uint8), (1, 1, 1))
    step = {"setup_dir": str(setup), "checkpoint": str(setup / "model_checkpoint_10"),
            "input_datasets": [raw], "output_datasets_prefix": str(tmp_path / "out.zarr"), "num_gpus": 1}
    (tmp_path / "predict.toml").write_text(toml.dumps({"predict": step}))

    run_prediction(str(tmp_path / "predict.toml"))

    out = capsys.readouterr()
    assert "'args'" not in out.out + out.err
    assert raw not in out.out + out.err


def test_segment_prints_no_config_dict(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(watershed_mutex, "mutex_watershed_segmentation", lambda config: None)
    step = {
        "affs_dataset": f"{tmp_path}/v.zarr/3d_affs",
        "fragments_dataset": f"{tmp_path}/o.zarr/frags",
        "seg_dataset_prefix": f"{tmp_path}/o.zarr/segs",
        "params": {"aff_neighborhood": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]], "bias": [-0.5] * 3},
    }

    run_segmentation(step, "mws")

    out = capsys.readouterr().out
    assert "'aff_neighborhood'" not in out
    assert out.startswith("defaults applied: ")


def segmentation_round(tmp_path, roots):
    """Labels of two halves, a 3d_affs prediction, and per root name a segmentation of one
    object plus its _fill_holes copy under segmentations_mws."""
    labels = np.ones((8, 32, 32), np.uint64)
    labels[:, :, 16:] = 2
    write(tmp_path / "v.zarr" / "labels", labels)
    write(tmp_path / "v.zarr" / "3d_affs", np.full((3, 8, 32, 32), 128, np.uint8))
    for root in roots:
        for leaf in (root, root + "_fill_holes"):
            write(tmp_path / "v.zarr" / "segmentations_mws" / leaf, np.ones((8, 32, 32), np.uint64))
    (tmp_path / "06_evaluate.toml").write_text(
        f'[volumes.v]\n[evaluate.v]\nseg_datasets_prefix = "{tmp_path}/v.zarr/segmentations_mws"\n'
        f'[evaluate.v.gt]\nlabels_dataset = "{tmp_path}/v.zarr/labels"\n'
        f'[evaluate.v.pred]\npred_dataset = "{tmp_path}/v.zarr/3d_affs"\n'
        "[evaluate.v.pred.params]\naff_neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]\n"
    )
    return str(tmp_path / "06_evaluate.toml")


def test_evaluate_ends_with_one_line_per_volume_and_segmentation(tmp_path, bs):
    root = "gb1_-0.5--eps0.001--b-0.4--rd64"
    config = segmentation_round(tmp_path, [root])

    r = bs("evaluate", config)

    assert r.returncode == 0, r.stderr
    gt, pred = f"({tmp_path}/06_evaluate_v_gt.json)", f"({tmp_path}/06_evaluate_v_pred.json)"
    # one object over two halves: every voxel pair of the other half is a merge
    assert r.stdout.splitlines()[-4:] == [
        f"v  mws             NVI split 0.000 merge 1.000  VOI split 0.00 merge 1.00  {gt}",
        f"v  mws/fill_holes  NVI split 0.000 merge 1.000  VOI split 0.00 merge 1.00  {gt}",
        f"v  mws             error mask 1.000 of the ROI                             {pred}",
        f"v  mws/fill_holes  error mask 1.000 of the ROI                             {pred}",
    ]
    for noise in ("Stats for", "{'", "] (", "Saving stats"):
        assert noise not in r.stdout + r.stderr
    results = json.loads((tmp_path / "06_evaluate_v_gt.json").read_text())
    assert list(results) == [f"{tmp_path}/v.zarr/segmentations_mws/{root}", f"{tmp_path}/v.zarr/segmentations_mws/{root}_fill_holes"]
    assert set(results[f"{tmp_path}/v.zarr/segmentations_mws/{root}"]) == {"report", "seg_ds", "labels_ds", "skeletons_file", "mask_ds", "metrics"}


def test_evaluate_names_a_segmentation_by_the_params_that_differ(tmp_path, bs):
    config = segmentation_round(tmp_path, ["gb1_-0.5--b-0.3--rd64", "gb1_-0.5--b-0.4--rd64"])

    r = bs("evaluate", config, "--gt")

    assert r.returncode == 0, r.stderr
    assert [line.split("  ")[1] for line in r.stdout.splitlines()[-4:]] == [
        "mws b-0.3", "mws b-0.3/fill_holes", "mws b-0.4", "mws b-0.4/fill_holes",
    ]


TRAIN = """
import sys
import pytorch_lightning as pl
import torch
from bootstrapper import training

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Linear(1, 1)
    def training_step(self, batch, i):
        return {"loss": self.w(batch["x"]).sum()}
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

class Ones(torch.utils.data.IterableDataset):
    def __iter__(self):
        while True:
            yield {"x": torch.ones(1, 1)}

training.fit(Model(), Ones(), sys.argv[1], 2, 1, pl.Callback(), num_workers=0)
print("matmul precision", torch.get_float32_matmul_precision())
"""


@pytest.mark.parametrize("resume", [False, True])
def test_training_prints_no_lightning_tip_and_no_setup_dir_warning(tmp_path, resume):
    setup = tmp_path / "setup"
    setup.mkdir()
    (setup / "train.py").write_text(TRAIN)
    run = [sys.executable, str(setup / "train.py"), str(setup)]
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    if resume:
        subprocess.run(run, env=env, capture_output=True, check=True, timeout=200)

    r = subprocess.run(run, env=env, capture_output=True, text=True, timeout=200)

    assert r.returncode == 0, r.stderr
    assert "matmul precision high" in r.stdout.splitlines()
    for noise in ("Tip:", "Checkpoint directory"):
        assert noise not in r.stdout + r.stderr
