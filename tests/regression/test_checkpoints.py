"""A fresh install has no checkpoint. bs prepare downloads each shipped checkpoint the round
reads, the dry run refuses one that is not there, and predict never prompts. An output key at
step level in a step over several volumes is refused, as every volume would write one path."""

import io
import shutil
import sys
import zipfile

import click
import numpy as np
import pytest
import requests
from click.testing import CliRunner
from test_prepare import write

from bootstrapper import configs, derive
from bootstrapper.config import load
from bootstrapper.predict import resolve_checkpoint
from bootstrapper.prepare import prepare

CLI, PREPARE, PREDICT = (sys.modules[f"bootstrapper.{m}"] for m in ("cli", "prepare", "predict"))
cli = CLI.cli
FIX = "run bs prepare round_01 again, or bs utils download-ckpts -m 3Af2M"
SHIPPED = "3d_affs_from_2d_mtlsd"


@pytest.fixture
def models(tmp_path, monkeypatch):
    """MODEL_DIR as a fresh install has it: the model scripts and no checkpoint."""
    models = tmp_path / "models"
    shutil.copytree(configs.MODEL_DIR, models, ignore=shutil.ignore_patterns("model_checkpoint_*", "__pycache__"))
    for module in (configs, derive, CLI, PREPARE, PREDICT):
        monkeypatch.setattr(module, "MODEL_DIR", str(models), raising=False)
    return models


@pytest.fixture
def network(monkeypatch):
    """requests.get serves each release zip with one 2 MB checkpoint; returns the urls asked."""
    asked = []

    class Zip:
        def __init__(self, url):
            model = url.rsplit("/", 1)[1].removesuffix(".zip")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                z.writestr(f"{model}/model_checkpoint_25000.ckpt", b"\0" * 2_000_000)
            self.data, self.headers = buf.getvalue(), {}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            return (self.data[i : i + chunk_size] for i in range(0, len(self.data), chunk_size))

    def get(url, **kwargs):
        asked.append(url)
        return Zip(url)

    monkeypatch.setattr(requests, "get", get)
    return asked


def volume(tmp_path, name):
    labels = np.zeros((8, 64, 64), np.uint64)
    labels[2, 10:30, 10:30] = 1
    labels[5, 30:50, 30:50] = 2
    write(tmp_path / f"{name}.zarr" / "raw", np.full(labels.shape, 100, np.uint8))
    write(tmp_path / f"{name}.zarr" / "labels", labels)
    return [f"{tmp_path}/{name}.zarr/raw", "--labels", "labels", "--labelled", "sparsely",
            "--untraced", "background", "--gt", "labels", "--name", name]


def bs_prepare(tmp_path, *names):
    """bs prepare round_01 with the first volume, then once per other volume."""
    out = []
    for i, name in enumerate(names):
        flags = ["--raw", *volume(tmp_path, name)] + ["--models", "2M,3Af2M"] * (i == 0)
        out.append(CliRunner().invoke(prepare, ["round_01", *flags]))
    return out


def test_bs_prepare_downloads_the_shipped_checkpoint_once(tmp_path, models, network):
    result = bs_prepare(tmp_path, "apical")[0]

    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.startswith("Download")]
    assert lines == [f"Downloaded the {SHIPPED} checkpoint (2 MB)"]
    assert network == [configs.MODEL_URLS[SHIPPED]]
    ckpt = models / SHIPPED / SHIPPED / "model_checkpoint_25000.ckpt"
    assert ckpt.stat().st_size == 2_000_000
    assert load("round_01/03_predict_3d_affs_from_2d_mtlsd.toml")[0].keys["checkpoint"] == str(ckpt)

    again = CliRunner().invoke(prepare, ["round_01"])

    assert again.exit_code == 0, again.output
    assert "Download" not in again.output
    assert len(network) == 1
    dry_run = CliRunner().invoke(cli, ["run", "round_01", "--dry-run"])
    assert dry_run.exit_code == 0, dry_run.output
    assert "model_checkpoint_25000.ckpt (shipped)" in dry_run.output
    assert dry_run.output.endswith("Every input exists or comes from an earlier step. Nothing ran.\n")


def test_a_network_error_stops_bs_prepare_with_the_fix(tmp_path, models, monkeypatch):
    def down(url, **kwargs):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "get", down)

    result = bs_prepare(tmp_path, "apical")[0]

    assert result.exit_code == 1
    assert "no route to host" in result.output
    assert result.output.rstrip().endswith(FIX), result.output
    assert not list(models.glob(f"{SHIPPED}/**/model_checkpoint_*"))


def test_the_dry_run_refuses_a_missing_shipped_checkpoint_before_any_step(tmp_path, models, network):
    """A round prepared before bs prepare downloaded: its tail names the checkpoint, and the
    file is not there."""
    fallback = models / SHIPPED / "model_checkpoint_25000"
    fallback.touch()
    assert bs_prepare(tmp_path, "apical")[0].exit_code == 0
    assert network == []
    fallback.unlink()

    for args in (["--dry-run"], []):
        result = CliRunner().invoke(cli, ["run", "round_01", *args])

        assert result.exit_code == 1, result.output
        assert "(shipped)" not in result.output
        assert result.stderr == (
            "Error: 03_predict_3d_affs_from_2d_mtlsd.toml [predict.3d_affs_from_2d_mtlsd] apical reads "
            f"{fallback}: the shipped checkpoint is not there. Nothing ran. To download it, {FIX}\n"
        )
    assert not (tmp_path / "round_01" / "runs").exists()

    CliRunner().invoke(prepare, ["round_01"])
    dry_run = CliRunner().invoke(cli, ["run", "round_01", "--dry-run"])

    assert dry_run.exit_code == 0, dry_run.output
    assert "model_checkpoint_25000.ckpt (shipped)" in dry_run.output


def test_predict_stops_at_once_on_a_missing_shipped_checkpoint(models, network):
    setup = models / SHIPPED

    with pytest.raises(click.ClickException) as e:
        resolve_checkpoint(str(setup), str(setup / "model_checkpoint_25000"))

    assert e.value.message.endswith("; run bs utils download-ckpts -m 3Af2M")
    assert network == []


OUTPUTS = {
    "segment": ("04_segment.toml", "[segment]", "seg_dataset_prefix", '"segs_try2"'),
    "predict": ("02_predict_2d_mtlsd.toml", "[predict.2d_mtlsd]", "output_datasets_prefix", '"preds"'),
    "refine": ("05_refine.toml", "[refine.fill-holes]", "out_array", '"filled"'),
    "evaluate": ("06_evaluate.toml", "[evaluate]", "out_dir", '"errors"'),
}


@pytest.mark.parametrize("kind", OUTPUTS)
def test_an_output_key_for_two_volumes_is_refused_with_the_fix(tmp_path, kind):
    file, table, key, value = OUTPUTS[kind]
    assert [r.exit_code for r in bs_prepare(tmp_path, "a", "b")] == [0, 0]
    path = tmp_path / "round_01" / file
    path.write_text(path.read_text().replace(f"{table}\n", f"{table}\n{key} = {value}\n", 1))

    prepared = CliRunner().invoke(prepare, ["round_01"])
    result = CliRunner().invoke(cli, ["run", "round_01", "--dry-run"])

    where = table[1:-1].split(".")
    step, op = (where[0], "." + where[1]) if kind == "refine" else (table[1:-1], "")
    fix = (f"{table} sets {key} for 2 volumes, so both write one path. "
           f"Set {key} under [{step}.a{op}] and [{step}.b{op}], one per volume.\n")
    assert result.exit_code == 1, result.output
    assert result.stderr.endswith(fix), result.stderr
    assert prepared.exit_code == 1
    assert prepared.stderr.endswith(fix), prepared.stderr


def test_an_output_key_for_one_volume_is_allowed(tmp_path):
    bs_prepare(tmp_path, "a")
    path = tmp_path / "round_01" / "04_segment.toml"
    path.write_text(path.read_text().replace("[segment]\n", '[segment]\nseg_dataset_prefix = "segs_try2"\n', 1))
    CliRunner().invoke(prepare, ["round_01"])

    result = CliRunner().invoke(cli, ["run", "round_01", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "segs_try2" in result.output
