"""bs run of a round skips the leading steps that are done: the newest receipt that ran each
says done, it read the keys it reads now, and what it writes is there. The step runners are
stubs, so no model runs."""

import glob
import os
import shutil
import subprocess
import sys
import time

import click
import pytest
from click.testing import CliRunner

import bootstrapper.cli as bs_cli
from bootstrapper.cli import cli
from bootstrapper.config import WRITES, load, paths, runs

BS = os.path.join(os.path.dirname(sys.executable), "bs")


@pytest.fixture
def round_dir(tmp_path):
    """01 predict, 02 segment, 03 evaluate; each reads what the step before writes."""
    (tmp_path / "raw.zarr" / "raw").mkdir(parents=True)
    out = f"{tmp_path}/out.zarr"
    path = tmp_path / "round"
    path.mkdir()
    (path / "01_predict_a.toml").write_text(
        f'[predict.a]\ninput_datasets = ["{tmp_path}/raw.zarr/raw"]\noutput_datasets_prefix = "{out}/a"\n'
    )
    (path / "02_segment.toml").write_text(
        f'[segment]\naffs_dataset = "{out}/a/affs"\nfragments_dataset = "{out}/frags"\n'
        f'lut_dir = "{out}/luts"\nseg_dataset_prefix = "{out}/segs"\n'
        "[segment.params]\nremove_debris = 64\n"
    )
    (path / "03_evaluate.toml").write_text(f'[evaluate]\nseg_datasets_prefix = "{out}/segs"\n')
    return path


@pytest.fixture
def ran(monkeypatch):
    """The kinds that ran, in order. Each stub step makes the arrays its step writes; predict
    writes affs under its prefix."""
    kinds = []

    def stub(kind):
        def run(config_file, step):
            kinds.append(kind)
            for s in load(config_file):
                for _, config in runs(s):
                    for p in (p for key in WRITES[kind] for p in paths(config, key)):
                        os.makedirs(f"{p}/affs" if kind == "predict" else p, exist_ok=True)
        return run

    for kind in ("predict", "segment", "evaluate"):
        monkeypatch.setattr(bs_cli, kind, stub(kind))
    return kinds


def bs_run(*args):
    time.sleep(1)  # one receipt per second: runs/<stamp>.toml
    return CliRunner().invoke(cli, ["run", *map(str, args)])


def test_a_round_that_ran_runs_nothing_again(round_dir, ran):
    assert bs_run(round_dir).exit_code == 0
    assert ran == ["predict", "segment", "evaluate"]

    result = bs_run(round_dir)

    assert result.exit_code == 0, result.output
    assert ran == ["predict", "segment", "evaluate"]
    assert result.output == (
        "01_predict_a.toml to 03_evaluate.toml are done and unchanged; nothing to run "
        f"(bs run {round_dir} --from 01 runs every step)\n"
    )


def test_an_edited_key_reruns_its_step_and_every_step_after_it(round_dir, ran):
    bs_run(round_dir)
    segment = round_dir / "02_segment.toml"
    segment.write_text(segment.read_text().replace("remove_debris = 64", "remove_debris = 100"))
    ran.clear()

    dry_run = bs_run(round_dir, "--dry-run")
    result = bs_run(round_dir)

    skip = (f"01_predict_a.toml is done and unchanged; running from 02_segment.toml "
            f"(bs run {round_dir} --from 01 runs every step)")
    assert dry_run.output.splitlines()[:3] == [skip, f"{round_dir}: 2 steps.", "02 segment"]
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == skip
    assert ran == ["segment", "evaluate"]


def test_a_deleted_output_reruns_its_step(round_dir, ran, tmp_path):
    bs_run(round_dir)
    shutil.rmtree(tmp_path / "out.zarr" / "luts")
    ran.clear()

    result = bs_run(round_dir)

    assert result.exit_code == 0, result.output
    assert ran == ["segment", "evaluate"]


def test_a_failed_step_reruns(round_dir, ran, monkeypatch):
    real = bs_cli.segment

    def fail(config_file, step):
        real(config_file, step)  # its arrays are there, but not all their blocks
        raise click.ClickException("the affinities are empty\n  every block is 0")

    monkeypatch.setattr(bs_cli, "segment", fail)
    assert bs_run(round_dir).exit_code == 1
    monkeypatch.setattr(bs_cli, "segment", real)
    ran.clear()

    result = bs_run(round_dir)

    assert result.exit_code == 0, result.output
    assert ran == ["segment", "evaluate"]


def test_a_step_stopped_while_it_ran_reruns(round_dir, ran, monkeypatch):
    """The older receipt says done with the same keys, but the stopped run may have overwritten
    part of the output."""
    bs_run(round_dir)

    def stop(config_file, step):
        raise KeyboardInterrupt

    real = bs_cli.segment
    monkeypatch.setattr(bs_cli, "segment", stop)
    assert bs_run(round_dir, "--from", "02").exit_code == 1
    monkeypatch.setattr(bs_cli, "segment", real)
    ran.clear()

    result = bs_run(round_dir)

    assert result.exit_code == 0, result.output
    assert ran == ["segment", "evaluate"]


def test_a_step_reruns_when_a_step_before_it_ran_again(round_dir, ran):
    bs_run(round_dir)
    bs_run(round_dir, "--only", "01")
    ran.clear()

    result = bs_run(round_dir)

    assert result.exit_code == 0, result.output
    assert ran == ["segment", "evaluate"]


def test_from_01_runs_every_step(round_dir, ran):
    bs_run(round_dir)
    ran.clear()

    result = bs_run(round_dir, "--from", "01")

    assert result.exit_code == 0, result.output
    assert ran == ["predict", "segment", "evaluate"]
    assert "done and unchanged" not in result.output


def test_a_closed_pipe_ends_quietly_with_no_log_file(round_dir, tmp_path):
    bs = subprocess.Popen([BS, "run", str(round_dir), "--dry-run"], cwd=tmp_path,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    bs.stdout.close()  # as head does after its lines

    stderr = bs.stderr.read()
    bs.wait(timeout=120)

    assert glob.glob(f"{tmp_path}/bs_error_*.log") == []
    assert stderr == ""
