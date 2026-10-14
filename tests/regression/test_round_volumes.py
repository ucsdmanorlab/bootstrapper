"""A round owns its outputs: predict, segment, refine and evaluate write into the round, never
into your containers. bs prepare adds one volume per command to a round, writes the step files
the volumes need, and says why it asks nothing without a terminal."""

import glob
import os

import numpy as np
import pytest
import toml
from click.testing import CliRunner
from test_prepare import write

from bootstrapper.cli import cli
from bootstrapper.config import WRITES, load_round, load_volumes, paths, runs
from bootstrapper.catalog import parse_chain
from bootstrapper.prepare import Answers, Draft, parse_gt, parse_raw, prepare, write_round


def data(tmp_path):
    """volume_1 with sparse labels on one section, volume_2 with dense labels, as in CREMI A."""
    sparse = np.zeros((1, 16, 16), np.uint64)
    sparse[0, 4:8, 4:8] = 1
    for v in ("volume_1", "volume_2"):
        write(tmp_path / f"{v}.zarr" / "raw", np.zeros((4, 16, 16), np.uint8))
        write(tmp_path / f"{v}.zarr" / "labels", np.ones((4, 16, 16), np.uint64))
    write(tmp_path / "volume_1.zarr" / "sparse_labels", sparse, offset=(50, 0, 0))


FIRST = ["round_01", "--raw", "volume_1.zarr/raw", "--labels", "sparse_labels", "--labelled", "sparsely"]


def cremi(tmp_path):
    """The README's first command, on the data."""
    data(tmp_path)
    return CliRunner().invoke(prepare, FIRST, catch_exceptions=False)


def tree(path):
    return sorted(glob.glob(f"{path}/**", recursive=True))


def written(round_dir):
    """Every path a step of the round writes, by volume."""
    return [p for _, steps in load_round(round_dir) for s in steps for _, config in runs(s)
            for key in WRITES[s.kind] for p in paths(config, key)]


def test_a_fresh_round_writes_every_output_in_the_round(tmp_path):
    data(tmp_path)
    before = tree(tmp_path)

    CliRunner().invoke(prepare, FIRST, catch_exceptions=False)
    CliRunner().invoke(prepare, ["round_01", "--raw", "volume_2.zarr/raw", "--gt", "labels"], catch_exceptions=False)

    d = f"{tmp_path}/round_01"
    predict = load_round(d)[1][1][0]
    assert [c["output_datasets_prefix"] for _, c in runs(predict)] == [f"{d}/volume_1.zarr/2d_mtlsd", f"{d}/volume_2.zarr/2d_mtlsd"]
    out = written(d)
    assert len(out) == 13 and all(p.startswith(d + "/") for p in out), out
    assert [p for p in tree(tmp_path) if not p.startswith(d)] == before


def test_the_outputs_key_puts_a_volume_s_outputs_where_it_says(tmp_path):
    cremi(tmp_path)
    volumes = tmp_path / "round_01" / "volumes.toml"
    volumes.write_text(volumes.read_text() + f'outputs = "{tmp_path}/scratch/v1.zarr"\n')

    CliRunner().invoke(prepare, ["round_01"], catch_exceptions=False)

    predict = load_round("round_01")[1][1][0]
    assert predict.volumes["volume_1-zarr"]["output_datasets_prefix"] == f"{tmp_path}/scratch/v1.zarr/2d_mtlsd"


def test_glob_volumes_get_the_names_the_summary_prints_and_their_own_outputs(tmp_path):
    labels = np.zeros((4, 16, 16), np.uint64)
    labels[1, 4:8, 4:8] = 1
    for c in ("a", "b", "c"):
        write(tmp_path / c / "v.zarr" / "raw", np.zeros(labels.shape, np.uint8))
        write(tmp_path / c / "v.zarr" / "labels", labels)

    out = CliRunner().invoke(prepare, ["round_01", "--raw", "*/v.zarr/raw", "--labels", "labels", "--labelled", "sparsely"],
                             catch_exceptions=False).output

    assert "3 volumes (v-zarr, v-zarr-2, v-zarr-3)" in out
    assert list(load_volumes("round_01/volumes.toml")) == ["v-zarr", "v-zarr-2", "v-zarr-3"]
    samples = toml.load("round_01/01_train_2d_mtlsd.toml")["run"]["train"]["2d_mtlsd"]["samples"]
    assert [s["volume"] for s in samples] == ["v-zarr", "v-zarr-2", "v-zarr-3"]
    predict = load_round("round_01")[1][1][0]
    assert [c["output_datasets_prefix"] for _, c in runs(predict)] == [
        f"{tmp_path}/round_01/{c}/2d_mtlsd" for c in ("v.zarr", "v-2.zarr", "v-3.zarr")]


def test_ground_truth_labels_names_the_array_when_the_volume_has_no_labels(tmp_path):
    write(tmp_path / "v.zarr" / "raw", np.zeros((4, 16, 16), np.uint8))
    write(tmp_path / "v.zarr" / "labels", np.ones((4, 16, 16), np.uint64))
    write(tmp_path / "v.zarr" / "sparse", np.ones((1, 16, 16), np.uint64))
    d = parse_raw(str(tmp_path / "v.zarr" / "raw"), Draft())

    assert parse_gt("labels", d).keys["gt"] == str(tmp_path / "v.zarr" / "labels")
    reuse = Draft(keys=d.keys | {"labels": str(tmp_path / "v.zarr" / "sparse")}, raw=d.raw, arrays=d.arrays)
    assert parse_gt("labels", reuse).keys["gt"] == str(tmp_path / "v.zarr" / "sparse")


def test_bs_prepare_adds_a_volume_and_the_evaluate_step_it_needs(tmp_path):
    first = cremi(tmp_path)
    own = (tmp_path / "round_01" / "volumes.toml").read_text()

    result = CliRunner().invoke(prepare, ["round_01", "--raw", "volume_2.zarr/raw", "--gt", "labels"], catch_exceptions=False)

    assert first.output.endswith("Next: bs run round_01\n"
                                 "bs run round_01 --dry-run checks every path and runs nothing.\n"
                                 "To change a param, edit its file; every key says what it does.\n")
    assert result.exit_code == 0, result.output
    assert result.output == (
        "  volume_2-zarr: raw, ground truth labels\n"
        "added volume_2-zarr to round_01/volumes.toml\n"
        "wrote round_01/06_evaluate.toml\n"
        "refreshed 02_predict_2d_mtlsd.toml: [run.predict.2d_mtlsd.volume_2-zarr] input_datasets, output_datasets_prefix, chain_str\n"
        "refreshed 03_predict_3d_affs_from_2d_mtlsd.toml: [run.predict.3d_affs_from_2d_mtlsd.volume_2-zarr] "
        "input_datasets, output_datasets_prefix, chain_str\n"
        "refreshed 04_segment.toml: [run.segment.volume_2-zarr] affs_dataset, fragments_dataset, lut_dir, seg_dataset_prefix\n"
        "refreshed 05_refine.toml: [run.refine.volume_2-zarr] in_array\n"
        "round_01 is current.\n"
        "Next: bs run round_01\n")
    assert (tmp_path / "round_01" / "volumes.toml").read_text().startswith(own)
    assert load_volumes("round_01/volumes.toml")["volume_2-zarr"].keys == {
        "raw": f"{tmp_path}/volume_2.zarr/raw", "gt": f"{tmp_path}/volume_2.zarr/labels"}
    evaluate = load_round("round_01")[-1][1][0]
    assert list(evaluate.volumes) == ["volume_2-zarr"]
    assert evaluate.volumes["volume_2-zarr"]["seg_datasets_prefix"].startswith(f"{tmp_path}/round_01/volume_2.zarr/")
    dry_run = CliRunner().invoke(cli, ["run", "round_01", "--dry-run"]).output
    assert dry_run.endswith("comes from an earlier step. Nothing ran.\n"), dry_run
    assert "06 evaluate on volume_2-zarr\n" in dry_run


def test_bs_prepare_writes_the_step_a_volume_added_by_hand_needs(tmp_path):
    cremi(tmp_path)
    volumes = tmp_path / "round_01" / "volumes.toml"
    volumes.write_text(volumes.read_text() + f'\n[volumes.v2]\nraw = "{tmp_path}/volume_2.zarr/raw"\n'
                                              f'gt = "{tmp_path}/volume_2.zarr/labels"\n')

    result = CliRunner().invoke(prepare, ["round_01"], catch_exceptions=False)

    assert result.output.startswith("wrote round_01/06_evaluate.toml\nrefreshed 02_predict_2d_mtlsd.toml: ")
    assert result.output.endswith("round_01 is current.\nNext: bs run round_01\n")
    assert list(load_round("round_01")[-1][1][0].volumes) == ["v2"]
    assert CliRunner().invoke(prepare, ["round_01"]).output == "round_01 is current.\nNext: bs run round_01\n"


@pytest.mark.parametrize("args, error", [
    (["--raw", "volume_1.zarr/raw", "--labels", "sparse_labels", "--labelled", "sparsely", "--raw", "volume_2.zarr/raw", "--gt", "labels"],
     "--raw: give one volume per command; add the next with bs prepare round_02 --raw volume_2.zarr/raw"),
    (["--raw", "volume_2.zarr/raw", "--gt", "labels", "--gt", "sparse_labels"],
     "--gt: give one volume per command; add the next with bs prepare round_02 --raw ... --gt sparse_labels"),
])
def test_a_repeated_flag_is_refused_with_the_command_for_the_next_volume(tmp_path, args, error):
    cremi(tmp_path)

    result = CliRunner().invoke(prepare, ["round_02", *args])

    assert result.stderr == f"Error: {error}\n"
    assert not os.path.exists("round_02")


@pytest.mark.parametrize("args, error", [
    (["--raw", "volume_1.zarr/raw"], "round_01 has a volume named volume_1-zarr already. Give this one another name, "
     "such as --name volume_1-zarr-2"),
    (["--raw", "volume_2.zarr/raw", "--name", "volume_1-zarr"], "round_01 has a volume named volume_1-zarr already. "
     "Give this one another name, such as --name volume_1-zarr-2"),
    (["--raw", "volume_2.zarr/raw", "--models", "3M"], "--models: round_01 is a round already. Edit its files, then run: "
     "bs prepare round_01"),
])
def test_adding_a_volume_under_a_name_the_round_has_is_refused(tmp_path, args, error):
    cremi(tmp_path)
    before = {f: open(f).read() for f in glob.glob("round_01/*.toml")}

    result = CliRunner().invoke(prepare, ["round_01", *args])

    assert result.stderr == f"Error: {error}\n"
    assert {f: open(f).read() for f in glob.glob("round_01/*.toml")} == before


def test_without_a_terminal_bs_prepare_says_why_it_asks_nothing(tmp_path):
    result = CliRunner().invoke(prepare, ["round_01"], input="volume_1.zarr\n")

    assert result.stderr == ("Error: stdin is not a terminal, so bs prepare asks nothing. Give the facts as flags, "
                             "such as --raw data.zarr/raw/s0\n")


def test_a_flag_error_puts_a_colon_after_the_flag(tmp_path):
    data(tmp_path)

    result = CliRunner().invoke(prepare, ["round_01", "--raw", "volume_2.zarr/raw", "--gt", "dense"])

    assert result.stderr == f"Error: --gt: dense: no zarr array there or in {tmp_path}/volume_2.zarr\n"


def test_a_held_out_volume_with_a_target_mask_gets_the_refine_mask_op(tmp_path):
    """As in harris15: the training volume has no target mask, and the held-out volume has its
    gt mask. mws leaves objects up to 27 voxels outside the mask; the refine mask op clears them."""
    data(tmp_path)
    write(tmp_path / "volume_2.zarr" / "labels_mask", np.ones((4, 16, 16), np.uint8))
    CliRunner().invoke(prepare, FIRST, catch_exceptions=False)

    result = CliRunner().invoke(prepare, ["round_01", "--raw", "volume_2.zarr/raw", "--gt", "labels",
                                          "--target-mask", "labels_mask"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "added [refine.mask] to round_01/05_refine.toml\n" in result.output
    refine = load_round("round_01")[4][1][0]
    assert list(refine.keys["ops"]) == ["fill-holes", "mask"]
    assert dict(runs(refine))["volume_2-zarr"]["ops"]["mask"] == {"mask_array": f"{tmp_path}/volume_2.zarr/labels_mask"}
    dry_run = CliRunner().invoke(cli, ["run", "round_01", "--dry-run"]).output
    assert dry_run.endswith("Nothing ran.\n"), dry_run
    assert "05 refine on volume_1-zarr: fill-holes\n" in dry_run
    assert "05 refine on volume_2-zarr: fill-holes, mask\n" in dry_run


def test_one_volume_with_a_target_mask_is_enough_for_the_refine_mask_op(tmp_path):
    """The dialogue gives both volumes at once; the first has no target mask."""
    data(tmp_path)
    write(tmp_path / "volume_2.zarr" / "labels_mask", np.ones((4, 16, 16), np.uint8))
    volumes = {"v1": {"raw": f"{tmp_path}/volume_1.zarr/raw", "labels": f"{tmp_path}/volume_1.zarr/sparse_labels",
                      "2d_labels": True, "training_mask": "labels"},
               "v2": {"raw": f"{tmp_path}/volume_2.zarr/raw", "target_mask": f"{tmp_path}/volume_2.zarr/labels_mask"}}

    write_round("round_01", Answers(volumes, tuple(parse_chain("2M 3Af2M"))))

    assert list(load_round("round_01")[4][1][0].keys["ops"]) == ["fill-holes", "mask"]
