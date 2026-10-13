"""CPU predict runs with an empty device grant. The dry run prints the round's paths relative to
the round. A round gains a volume without a target mask, and only the masked volumes run the
mask op. A glob added to a round prints the volume names that the round gives."""

import subprocess

import numpy as np
import toml
from click.testing import CliRunner
from funlib.persistence import open_ds
from test_predict import make_config
from test_prepare import run_prepare, write

from bootstrapper.cli import cli
from bootstrapper.config import load, load_volumes, runs
from bootstrapper.predict import run_prediction
from bootstrapper.prepare import prepare


def test_cpu_predict_runs_with_an_empty_device_grant(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs: seen.append(kwargs.get("env") or {}))
    config_file = tmp_path / "predict.toml"
    config_file.write_text(toml.dumps({"predict": make_config(tmp_path, num_gpus=0)}))

    run_prediction(str(config_file))

    assert [env.get("CUDA_VISIBLE_DEVICES") for env in seen] == [""]


def test_a_dry_run_prints_round_paths_relative_to_the_round(tmp_path):
    run_prepare(tmp_path)

    lines = CliRunner().invoke(cli, ["run", "round_01", "--dry-run"]).output.splitlines()

    assert lines[0] == "round_01: 6 steps on 1 volume. Paths are in round_01 unless absolute."
    assert lines[1:4] == [
        "01 train.2d_mtlsd",
        f"   reads  {tmp_path}/apical.zarr/raw, {tmp_path}/apical.zarr/labels, training_mask.zarr/apical",
        "   writes 2d_mtlsd",
    ]
    assert lines[5] == "   reads  " f"{tmp_path}/apical.zarr/raw, 2d_mtlsd/model_checkpoint_30000.ckpt from 01"
    assert lines[6] == "   writes apical.zarr/2d_mtlsd"


def test_a_volume_without_a_target_mask_skips_the_mask_op(tmp_path):
    mask = write(tmp_path / "apical.zarr" / "mask", np.ones((8, 32, 48), np.uint8), offset=(200, 256, 256))
    run_prepare(tmp_path, mask=mask)
    write(tmp_path / "other.zarr" / "raw", np.full((8, 64, 64), 100, np.uint8))
    CliRunner().invoke(prepare, ["round_01", "--raw", "other.zarr/raw", "--name", "other"], catch_exceptions=False)
    path = tmp_path / "round_01" / "05_refine.toml"
    [refine] = load(str(path))
    segs = {vol: write(c["in_array"], np.full((8, 64, 64), 7, np.uint64)) for vol, c in runs(refine)}

    dry_run = CliRunner().invoke(cli, ["run", "round_01", "--dry-run"]).output
    result = CliRunner().invoke(cli, ["run", str(path)])

    assert "05 refine on apical: fill-holes, mask\n" in dry_run
    assert "05 refine on other: fill-holes\n" in dry_run
    assert result.exit_code == 0, result.output + result.stderr
    masked = open_ds(f"{segs['apical']}_fill_holes_masked")
    assert masked[masked.roi].sum() == 7 * 4 * 32 * 32
    filled = open_ds(f"{segs['other']}_fill_holes")
    assert filled[filled.roi].sum() == 7 * 8 * 64 * 64


def test_a_mask_op_that_no_volume_can_run_is_refused(tmp_path):
    seg = write(tmp_path / "seg.zarr" / "seg", np.full((8, 64, 64), 7, np.uint64))
    (tmp_path / "refine.toml").write_text(f'[refine]\nin_array = "{seg}"\n[refine.mask]\n')

    result = CliRunner().invoke(cli, ["run", "refine.toml"])

    assert result.stderr == "Error: [refine.mask] needs mask_array, the mask to keep the labels inside\n"


def test_a_glob_added_to_a_round_prints_the_names_the_round_gives(tmp_path):
    labels = np.zeros((4, 16, 16), np.uint64)
    labels[1, 4:8, 4:8] = 1
    write(tmp_path / "apical.zarr" / "raw", np.zeros(labels.shape, np.uint8))
    write(tmp_path / "apical.zarr" / "labels", labels)
    for c in ("apical", "basal"):
        write(tmp_path / "more" / f"{c}.zarr" / "raw", np.zeros(labels.shape, np.uint8))
    CliRunner().invoke(prepare, ["round_01", "--raw", "apical.zarr/raw", "--labels", "labels", "--labelled", "sparsely"],
                       catch_exceptions=False)

    out = CliRunner().invoke(prepare, ["round_01", "--raw", "more/*.zarr/raw"], catch_exceptions=False).output

    assert list(load_volumes("round_01/volumes.toml")) == ["apical-zarr", "apical-zarr-2", "basal-zarr"]
    assert "2 volumes (apical-zarr-2, basal-zarr)" in out
