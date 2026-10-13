"""The command line: one spelling rule, one-line errors with the traceback in a log file,
and bs run of a file or a round directory."""

import glob
import os
import time

import click
import daisy.logging
import pytest
import toml
from click.testing import CliRunner

from bootstrapper.blockwise import check_task_states
from bootstrapper.cli import cli


def commands(cmd, path="bs"):
    yield path, cmd
    for name, sub in getattr(cmd, "commands", {}).items():
        yield from commands(sub, f"{path} {name}")


def test_spellings_follow_one_rule():
    """No underscore in a command or long option; -i is an input and -w the worker count, everywhere."""
    problems = []
    for path, cmd in commands(cli):
        if "_" in path:
            problems.append(f"{path}: command name with underscore")
        for p in cmd.params:
            if not isinstance(p, click.Option):
                continue
            opts = p.opts + p.secondary_opts
            shorts = [o for o in opts if not o.startswith("--")]
            problems += [
                f"{path}: {o}" for o in opts if o.startswith("--") and "_" in o
            ]
            if "-i" in shorts and not p.name.startswith("in_"):
                problems.append(f"{path}: -i means {p.name}")
            if {"-w", "-nw"} & set(shorts) and p.name != "num_workers":
                problems.append(f"{path}: {shorts} means {p.name}")
            if p.name == "num_workers" and shorts != ["-w"]:
                problems.append(f"{path}: num_workers is {shorts}")

    assert problems == []


def test_a_failure_prints_one_line_and_logs_the_traceback(tmp_path, bs):
    config = tmp_path / "04_eval.toml"
    config.write_text(
        '[evaluate]\nthresholds = [0.1, 1.0]\n[evaluate.pred]\npred_dataset = "/nowhere/3d_affs"\n'
    )

    r = bs("evaluate", config, "-p")

    assert r.returncode == 1
    assert "Error:" in r.stderr and "Traceback" not in r.stderr + r.stdout
    (log,) = glob.glob(f"{tmp_path}/bs_error_*.log")
    assert os.path.basename(log) in r.stderr
    stamp, command, *rest = open(log).read().splitlines()
    assert stamp.startswith(time.strftime("%Y%m%d"))
    assert command.startswith("$ ") and " evaluate " in command
    assert any(line.startswith("Traceback") for line in rest)


def test_click_usage_errors_print_as_before(tmp_path):
    result = CliRunner().invoke(cli, ["refine", "size-filter", "-i", "/nowhere"])

    assert result.exit_code == 2
    assert "does not exist" in result.stderr
    assert glob.glob(f"{tmp_path}/bs_error_*.log") == []


def test_failed_blocks_name_the_worker_log_dir():
    class TaskState:
        failed_count, orphaned_count, total_block_count = 2, 0, 9

    with pytest.raises(RuntimeError) as error:
        check_task_states({"Morph": TaskState()})

    assert "2 failed" in str(error.value)
    assert str(daisy.logging.get_log_basedir()) in str(error.value)


@pytest.fixture
def round_dir(tmp_path, affs):
    """A round directory holding 01_segment.toml: one-block mws of the affs."""
    step = {
        "affs_dataset": affs,
        "fragments_dataset": f"{tmp_path}/o.zarr/frags",
        "seg_dataset_prefix": f"{tmp_path}/o.zarr/segs",
        "lut_dir": f"{tmp_path}/o.zarr/luts",
        "method": "mws",
        "blockwise": False,
        "params": {
            "aff_neighborhood": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            "bias": [-0.5] * 3,
            "strides": [[1, 1, 1]] * 3,
            "randomized_strides": False,
            "noise_eps": 0.001,
            "filter_fragments": 0.0,
            "remove_debris": 0,
        },
    }
    path = tmp_path / "round"
    path.mkdir()
    (path / "01_segment.toml").write_text(toml.dumps({"segment": step}))
    return path


def test_run_a_round_by_file_prefix_then_chain_the_refine_ops(tmp_path, round_dir, bs):
    r = bs("run", round_dir, "--only", "01")

    assert r.returncode == 0, r.stderr
    assert "Running segment" in r.stdout
    (seg,) = glob.glob(f"{tmp_path}/o.zarr/segs/*")

    (round_dir / "02_refine.toml").write_text(
        f'[refine]\nin_array = "{seg}"\n'
        "[refine.fill-holes]\nxy = true\nnum_workers = 2\n"
        "[refine.size-filter]\nmin_size = 4\nnum_workers = 2\n"
    )
    r = bs("run", round_dir, "--from", "02")

    assert r.returncode == 0, r.stderr
    assert glob.glob(f"{seg}_fill_holes_size_filtered/.zarray")
    assert "Running segment" not in r.stdout

    r = bs("run", round_dir / "02_refine.toml", "--only", "refine")

    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "option,step,needles",
    [
        ("--only", "07", ["no such step"]),
        ("--from", "evaluate", ["steps are", "segment"]),
    ],
)
def test_run_refuses_an_unknown_step_by_name(round_dir, option, step, needles):
    result = CliRunner().invoke(cli, ["run", str(round_dir), option, step])

    assert result.exit_code != 0
    for needle in needles:
        assert needle in result.stderr


def test_run_refuses_a_directory_without_numbered_files(tmp_path):
    (tmp_path / "empty").mkdir()

    result = CliRunner().invoke(cli, ["run", str(tmp_path / "empty")])

    assert result.exit_code != 0
    assert "no numbered" in result.stderr
