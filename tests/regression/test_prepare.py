"""bs prepare asks the facts, or takes them as flags, then writes volumes.toml, numbered step files, a
checkpoint-free setup with no __pycache__, and the training masks in the round, never in the
data container; every file loads, and every param in it says what it does. Each file holds
your keys above a run tail of derived keys, which bs prepare PATH writes again from them.
bs run refuses a stale tail, checks every path before any step, and keeps a receipt."""

import glob
import json
import os
import re
import sys

import numpy as np
import pytest
import toml
from click.testing import CliRunner
from funlib.persistence import open_ds, prepare_ds

from bootstrapper import configs
from bootstrapper.catalog import parse_chain
from bootstrapper.cli import cli
from bootstrapper.config import load, load_round, runs
from bootstrapper.params import KEYS, defaults
from bootstrapper.prepare import Answers, prepare, write_round

PREPARE = sys.modules["bootstrapper.prepare"]

FILES = [
    "01_train_2d_mtlsd.toml",
    "02_predict_2d_mtlsd.toml",
    "03_predict_3d_affs_from_2d_mtlsd.toml",
    "04_segment.toml",
    "05_refine.toml",
    "06_evaluate.toml",
]


def write(path, data, chunk_shape=None, offset=(0, 0, 0), voxel_size=(50, 8, 8)):
    ds = prepare_ds(
        str(path),
        shape=data.shape,
        offset=offset,
        voxel_size=voxel_size,
        axis_names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=data.dtype,
        chunk_shape=chunk_shape,
    )
    ds[ds.roi] = data
    return str(path)


def run_prepare(tmp_path, models="", mask="", path="round_01"):
    """One volume, sparse labels on two sections, background beside each object."""
    labels = np.zeros((8, 64, 64), np.uint64)
    labels[2, 10:30, 10:30] = 1
    labels[5, 30:50, 30:50] = 2
    write(tmp_path / "apical.zarr" / "raw", np.full(labels.shape, 100, np.uint8))
    write(tmp_path / "apical.zarr" / "labels", labels)
    flags = ["--raw", f"{tmp_path}/apical.zarr/raw", "--labels", "labels", "--labelled", "sparsely",
             "--untraced", "background", "--gt", "labels", "--name", "apical"]
    flags += ["--target-mask", mask] * bool(mask) + ["--models", models] * bool(models)

    return CliRunner().invoke(prepare, [f"{tmp_path}/{path}", *flags], catch_exceptions=False)


@pytest.fixture
def prepared(tmp_path):
    return run_prepare(tmp_path)


def data(tmp_path):
    """A container with the raw at two scales, labels on one section, and a mask at two scales."""
    labels = np.zeros((8, 64, 64), np.uint64)
    labels[2, 10:30, 10:30] = 1
    write(tmp_path / "apical.zarr" / "raw" / "s0", np.zeros(labels.shape, np.uint8))
    write(tmp_path / "apical.zarr" / "raw" / "s1", np.zeros((8, 32, 32), np.uint8), voxel_size=(50, 16, 16))
    write(tmp_path / "apical.zarr" / "labels", labels)
    write(tmp_path / "apical.zarr" / "mask" / "s0", np.ones(labels.shape, np.uint8))
    write(tmp_path / "apical.zarr" / "mask" / "s1", np.ones((8, 32, 32), np.uint8), voxel_size=(50, 16, 16))


def step(tmp_path, name):
    return load(f"{tmp_path}/round_01/{name}")[0]


def test_the_flags_line_writes_the_files_in_run_order(tmp_path, prepared):
    assert prepared.exit_code == 0, prepared.output
    assert prepared.output.splitlines()[:3] == [
        "  labels: labels on 2 of 8 sampled sections, 0 with labels on the next section, so 2D labels",
        "  apical: raw, sparsely labelled labels (2D; background around objects), ground truth labels",
        "  2d_mtlsd → 3d_affs_from_2d_mtlsd",
    ]
    files = sorted(
        os.path.basename(p) for p in glob.glob(f"{tmp_path}/round_01/*.toml")
    )
    assert files == FILES + ["volumes.toml"]
    assert prepared.output.count("wrote ") == 7
    assert "bs run" in prepared.output
    for name in FILES:
        assert load(f"{tmp_path}/round_01/{name}")


def test_each_param_says_what_it_does_and_each_example_loads(tmp_path, prepared):
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


def test_the_volume_facts_are_recorded(tmp_path, prepared):
    volume = toml.load(f"{tmp_path}/round_01/volumes.toml")["volumes"]["apical"]

    assert volume == {
        "raw": f"{tmp_path}/apical.zarr/raw",
        "labels": f"{tmp_path}/apical.zarr/labels",
        "training_mask": "labels",
        "2d_labels": True,
        "mask_dilation": 10,
        "gt": f"{tmp_path}/apical.zarr/labels",
    }


def test_the_dialogue_explains_goes_back_and_asks_again(tmp_path, monkeypatch):
    """'?' explains a question, '<' goes back one answer, and a wrong answer is asked again
    with its fix."""
    data(tmp_path)
    monkeypatch.setattr(PREPARE, "_terminal", lambda: True)
    answers = ["apical.zarr", "1", "1", "1", "?", "b", "mask/s1", "mask/s0", "", "<", "labels", "segment", "", "", ""]

    result = CliRunner().invoke(prepare, ["round_01"], input="\n".join(answers) + "\n", catch_exceptions=False)

    assert result.output.split("\nCopied ")[0] == """\
Set up a round: your arrays, then the models. '?' explains a question, '<' goes back, Tab completes.
Raw image (zarr array or container, tif, image folder; globs work) > apical.zarr
    1  raw/s0  8 64 64  uint8  50 8 8 nm
    2  raw/s1  8 32 32  uint8  50 16 16 nm
Raw array > 1
    1  labels  8 64 64  uint64  50 8 8 nm
Labels (Enter: none, only predict) > 1
  labels: labels on 1 of 8 sampled sections, 0 with labels on the next section, so 2D labels
    1  sparsely labelled
    2  densely labelled
    3  densely labelled within a mask
Labelling > 1
Untraced space next to a traced object: [o]ther objects / [b]ackground [o] > ?
  other objects: as in EM neuropil, where untraced cells touch the traced ones
  background: as in cells on an empty slide; training learns 10 voxels of it around each object
Untraced space next to a traced object: [o]ther objects / [b]ackground [o] > b
  1 mask at 50 8 8 nm, such as mask/s0. Tab completes.
Target mask (Enter: none, segment everywhere) > mask/s1
  mask/s1 has voxel 50 16 16 nm; the raw has 50 8 8 nm. Use mask/s0.
Target mask (Enter: none, segment everywhere) > mask/s0
Ground truth ('labels' reuses the labels; Enter: none) > 
Volume name [apical-zarr] > <
Ground truth ('labels' reuses the labels; Enter: none) > labels
Volume name [apical-zarr] > segment
  a volume cannot be named train, predict, segment, refine, evaluate, volumes, run or round. Example: segment-1
Volume name [apical-zarr] > 
  apical-zarr: raw/s0, sparsely labelled labels (2D; background around objects), target mask mask/s0, ground truth labels
Another volume? [n] > 
Models (? lists all 9) [2d_mtlsd 3d_affs_from_2d_mtlsd] > 
  2d_mtlsd → 3d_affs_from_2d_mtlsd"""
    c = f"{tmp_path}/apical.zarr"
    assert toml.load("round_01/volumes.toml")["volumes"] == {"apical-zarr": {
        "raw": f"{c}/raw/s0", "labels": f"{c}/labels", "2d_labels": True, "training_mask": "labels",
        "mask_dilation": 10, "target_mask": f"{c}/mask/s0", "gt": f"{c}/labels"}}


@pytest.mark.parametrize("args, error", [
    (["round_01", "--raw", "apical.zarr"], "--raw {}/apical.zarr holds 2 image arrays; give one: raw/s0 (50 8 8 nm), "
     "raw/s1 (50 16 16 nm). Example: --raw {}/apical.zarr/raw/s0"),
    (["round_01", "--raw", "apical.zarr/raw/s0", "--labels", "labels"], "--labelled is needed with --labels: sparsely, "
     "densely, or a mask path (densely labelled within it). Example: --labelled sparsely"),
    (["round_01", "--raw", "apical.zarr/raw/s0", "--labels", "labels", "--labelled", "densely", "--target-mask", "mask/s1"],
     "--target-mask: mask/s1 has voxel 50 16 16 nm; the raw has 50 8 8 nm. Use mask/s0."),
    (["round_01", "--raw", "apical.zarr/raw/s0"], "no volume has labels, so no model can train. Give --labels; "
     "label arrays at 50 8 8 nm in apical.zarr: labels"),
    (["apical.zarr"], "apical.zarr is zarr data, not a round. Give a new round folder and the raw: "
     "bs prepare round_01 --raw apical.zarr"),
])
def test_without_a_terminal_a_missing_or_wrong_flag_names_its_fix(tmp_path, args, error):
    data(tmp_path)

    result = CliRunner().invoke(prepare, args)

    assert result.stderr == f"Error: {error.format(tmp_path, tmp_path)}\n"
    assert not os.path.exists("round_01")


@pytest.mark.parametrize("sections, labelled, labels_2d", [(1, [0], True), (8, [0, 2, 4, 6], True), (8, [3, 4], False)])
def test_labels_are_2d_unless_a_labelled_section_has_a_labelled_next_one(tmp_path, sections, labelled, labels_2d):
    """2D labels get 2d_labels = true and the default models for them, which train 2d_mtlsd."""
    labels = np.zeros((sections, 16, 16), np.uint64)
    labels[labelled, 4:8, 4:8] = 1
    raw = write(tmp_path / "v.zarr" / "raw", np.zeros(labels.shape, np.uint8))
    write(tmp_path / "v.zarr" / "labels", labels)

    CliRunner().invoke(prepare, ["round_01", "--raw", raw, "--labels", "labels", "--labelled", "densely"], catch_exceptions=False)

    assert toml.load("round_01/volumes.toml")["volumes"]["v-zarr"]["2d_labels"] is labels_2d
    assert os.path.exists(f"round_01/01_train_{'2d' if labels_2d else '3d'}_mtlsd.toml")


def test_a_toml_path_gets_the_round_in_one_file(tmp_path):
    result = run_prepare(tmp_path, path="round.toml")

    assert f"wrote {tmp_path}/round.toml\n" in result.output
    assert os.path.isfile("round.toml")
    assert [s.label for _, steps in load_round("round.toml") for s in steps] == [
        "train.2d_mtlsd", "predict.2d_mtlsd", "predict.3d_affs_from_2d_mtlsd", "segment", "refine", "evaluate"]
    dry_run = CliRunner().invoke(cli, ["run", "round.toml", "--dry-run"]).output
    assert dry_run.endswith("comes from an earlier step. Nothing ran.\n"), dry_run


def test_a_glob_or_your_own_volumes_toml_makes_several_volumes(tmp_path):
    """A glob of containers writes one volume table of globs. On a folder with your own
    volumes.toml, bs prepare keeps it and writes the steps."""
    labels = np.zeros((8, 16, 16), np.uint64)
    labels[2:5, 4:8, 4:8] = 1
    for c in ("a", "b"):
        write(tmp_path / f"{c}.zarr" / "raw", np.zeros(labels.shape, np.uint8))
        write(tmp_path / f"{c}.zarr" / "labels", labels)
    CliRunner().invoke(prepare, ["round_01", "--raw", "*.zarr", "--labels", "labels", "--labelled", "densely"], catch_exceptions=False)
    own = "# mine\n" + (tmp_path / "round_01" / "volumes.toml").read_text()
    (tmp_path / "round_02").mkdir()
    (tmp_path / "round_02" / "volumes.toml").write_text(own)

    CliRunner().invoke(prepare, ["round_02"], catch_exceptions=False)

    assert [v["labels"] for v in toml.load("round_01/volumes.toml")["volumes"].values()] == [f"{tmp_path}/*.zarr/labels"]
    assert (tmp_path / "round_02" / "volumes.toml").read_text() == own
    for name in ("round_01", "round_02"):
        samples = load(f"{name}/01_train_3d_mtlsd.toml")[0].keys["samples"]
        assert [(s["volume"], s["labels"]) for s in samples] == [("a-zarr", f"{tmp_path}/a.zarr/labels"), ("b-zarr", f"{tmp_path}/b.zarr/labels")]


def test_the_models_key_of_your_own_volumes_toml_picks_the_chain(tmp_path):
    """bs prepare --tui and the line dialogue write volumes.toml with a models key; bs prepare
    on that folder writes the steps of those models, not the default chain."""
    run_prepare(tmp_path)
    own = 'models = "2M 3Af2A"\n' + (tmp_path / "round_01" / "volumes.toml").read_text()
    (tmp_path / "round_02").mkdir()
    (tmp_path / "round_02" / "volumes.toml").write_text(own)

    result = CliRunner().invoke(prepare, [f"{tmp_path}/round_02"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert sorted(os.path.basename(p) for p in glob.glob(f"{tmp_path}/round_02/0*.toml"))[:3] == [
        "01_train_2d_mtlsd.toml", "02_predict_2d_mtlsd.toml", "03_predict_3d_affs_from_2d_affs.toml"]


def test_your_keys_sit_above_the_run_tail(tmp_path, prepared):
    """The head holds the params; every derived key sits in a [run.*] table below them."""
    heads = {
        "01_train_2d_mtlsd.toml": {"train": {"2d_mtlsd": defaults("train.*")}},
        "02_predict_2d_mtlsd.toml": {"predict": {"2d_mtlsd": defaults("predict.*")}},
        "03_predict_3d_affs_from_2d_mtlsd.toml": {
            "predict": {"3d_affs_from_2d_mtlsd": defaults("predict.*")}
        },
        "04_segment.toml": {
            "segment": defaults("segment") | {"params": defaults("segment.params")}
        },
        "05_refine.toml": {"refine": {"fill-holes": defaults("refine.fill-holes")}},
        "06_evaluate.toml": {"evaluate": defaults("evaluate")},
    }
    for name, head in heads.items():
        doc = toml.load(tmp_path / "round_01" / name)
        assert doc.pop("run"), name
        assert doc == head, name


def test_bs_prepare_writes_the_tails_again_and_keeps_your_lines(tmp_path, prepared):
    """A key you set leaves the tail, and what follows from it follows; the checkpoint follows
    max_iterations and the training mask its keys. Your lines stay byte for byte, an edited
    setup stays, and a second run changes nothing."""
    round_dir = tmp_path / "round_01"
    edits = {
        "01_train_2d_mtlsd.toml": ("= 30001", "= 20001  # mine"),
        "03_predict_3d_affs_from_2d_mtlsd.toml": (
            "]\n",
            ']\ncheckpoint = "/m/model_checkpoint_5000.ckpt"\n',
        ),
        "volumes.toml": ("mask_dilation = 10", "mask_dilation = 1"),
    }
    for name, (old, new) in edits.items():
        path = round_dir / name
        path.write_text(path.read_text().replace(old, new, 1))
    net_config = round_dir / "2d_mtlsd" / "net_config.json"
    net_config.write_text(net_config.read_text() + "\n")
    before = {f: f.read_text() for f in round_dir.glob("*.toml")}

    # a relative path, from tmp_path
    result = CliRunner().invoke(prepare, ["round_01"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    refreshed = "refreshed 02_predict_2d_mtlsd.toml: [run.predict.2d_mtlsd] checkpoint"
    assert f"{refreshed}\n" in result.output
    after = {f: f.read_text() for f in round_dir.glob("*.toml")}
    for f, text in before.items():
        assert after[f].startswith(text.split("\n# Below: the run tail.")[0]), f
    predict = round_dir / "02_predict_2d_mtlsd.toml"
    # the refreshed tail names the round as this run typed it
    assert after[predict] == before[predict].replace("_30000.ckpt", "_20000.ckpt").replace(f"prepare {round_dir}", "prepare round_01")
    third = toml.load(round_dir / "03_predict_3d_affs_from_2d_mtlsd.toml")
    assert "checkpoint" not in third["run"]["predict"]["3d_affs_from_2d_mtlsd"]
    affs = step(tmp_path, "04_segment.toml").volumes["apical"]["affs_dataset"]
    assert affs.endswith("/5000--from--2d_mtlsd_20000/3d_affs")
    mask = open_ds(f"{round_dir}/training_mask.zarr/apical")
    # each square of 20 x 20 grown 1 voxel in xy
    assert mask[mask.roi].sum() == 2 * (20 * 20 + 4 * 20)
    assert net_config.read_text().endswith("}\n\n")
    again = CliRunner().invoke(prepare, ["round_01"])
    assert again.output.endswith("training_mask.zarr/apical\nround_01 is current.\nNext: bs run round_01\n")
    assert {f: f.read_text() for f in round_dir.glob("*.toml")} == after
    flags = CliRunner().invoke(prepare, ["round_01", "--gt", "labels"])
    assert flags.stderr == ("Error: --raw is needed to add a volume to round_01. To change a volume, edit "
                            "round_01/volumes.toml, then run: bs prepare round_01\n")


def test_predict_covers_the_target_mask_box_and_refine_masks_to_it(tmp_path):
    """The box is cut to the raw. Segment, refine and evaluate read the mask itself."""
    mask = np.ones((8, 32, 48), np.uint8)
    mask = write(tmp_path / "apical.zarr" / "mask", mask, offset=(200, 256, 256))
    run_prepare(tmp_path, mask=mask)

    for name in ("02_predict_2d_mtlsd.toml", "03_predict_3d_affs_from_2d_mtlsd.toml"):
        apical = step(tmp_path, name).volumes["apical"]
        assert apical["roi_offset"] == apical["roi_shape"] == [200, 256, 256]
    assert step(tmp_path, "04_segment.toml").volumes["apical"]["mask_dataset"] == mask
    assert step(tmp_path, "06_evaluate.toml").volumes["apical"]["mask_dataset"] == mask
    [(_, refine)] = runs(step(tmp_path, "05_refine.toml"))
    assert refine["ops"] == {"fill-holes": {"xy": True}, "mask": {"mask_array": mask}}

    seg = write(refine["in_array"], np.full((8, 64, 64), 7, np.uint64))
    path = tmp_path / "round_01" / "05_refine.toml"
    path.write_text("[refine]\nnum_workers = 1\n" + path.read_text())
    result = CliRunner().invoke(cli, ["run", str(path)], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    masked = open_ds(f"{seg}_fill_holes_masked")
    assert masked[masked.roi].sum() == 7 * 4 * 32 * 32


def test_bs_prepare_writes_the_training_mask_from_the_labels_as_they_are_now(tmp_path, prepared):
    """A labels edit at the same path keys nothing new; the mask follows the labels all the same."""
    round_dir = tmp_path / "round_01"
    volumes = round_dir / "volumes.toml"
    volumes.write_text(volumes.read_text().replace("mask_dilation = 10", "mask_dilation = 1"))
    CliRunner().invoke(prepare, ["round_01"], catch_exceptions=False)
    labels = open_ds(f"{tmp_path}/apical.zarr/labels", "r+")
    edited = labels[labels.roi]
    edited[5] = 0
    edited[2, 40:50, 5:15] = 3
    labels[labels.roi] = edited

    result = CliRunner().invoke(prepare, ["round_01"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    fg = edited > 0
    grown = fg | np.any([np.roll(fg, s, a) for a in (1, 2) for s in (1, -1)], 0)
    mask = open_ds(f"{round_dir}/training_mask.zarr/apical")
    assert np.array_equal(mask[mask.roi], grown)


def test_a_fresh_train_step_trains_on_one_gpu(tmp_path, prepared):
    assert step(tmp_path, "01_train_2d_mtlsd.toml").keys["devices"] == 1


def test_the_train_step_trains_a_fresh_setup_on_a_mask_in_the_round(tmp_path, prepared):
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
    write_round(f"{tmp_path}/round_01", Answers(volumes, parse_chain("3M")))

    samples = step(tmp_path, "01_train_3d_mtlsd.toml").keys["samples"]
    for (name, (_, expected)), sample in zip(cases.items(), samples):
        # no mask: the trainer takes labels > 0
        mask = open_ds(sample["mask"]) if "mask" in sample else None
        trained_on = mask[mask.roi] if mask else fg
        assert np.array_equal(trained_on, expected), name
    assert sorted(glob.glob(f"{tmp_path}/data.zarr/**", recursive=True)) == container


def test_the_second_predict_step_reads_the_first_ones_output(tmp_path, prepared):
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
        f"{tmp_path}/round_01/apical.zarr/2d_mtlsd/30000/2d_lsds"
    ]


def test_segment_refine_and_evaluate_follow_the_affs(tmp_path, prepared):
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


def test_bs_prepare_names_the_round_and_the_scripts_as_you_typed_them(tmp_path):
    labels = np.zeros((8, 64, 64), np.uint64)
    labels[2, 10:30, 10:30] = 1
    write(tmp_path / "apical.zarr" / "raw", np.zeros(labels.shape, np.uint8))
    write(tmp_path / "apical.zarr" / "labels", labels)

    result = CliRunner().invoke(prepare, ["round_01", "--raw", "apical.zarr/raw", "--labels", "labels", "--labelled",
                                          "sparsely", "--untraced", "background"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "Copied the 2d_mtlsd scripts into round_01/2d_mtlsd" in result.output.splitlines()
    tail = "# Edit the keys above, not these; then run: bs prepare round_01"
    assert tail in (tmp_path / "round_01" / "01_train_2d_mtlsd.toml").read_text().splitlines()


def test_model_copy_leaves_out_pycache(tmp_path, monkeypatch):
    src = tmp_path / "bs" / "models" / "m"
    (src / "__pycache__").mkdir(parents=True)
    (src / "train.py").write_text("")
    (src / "__pycache__" / "train.cpython-312.pyc").write_text("")
    monkeypatch.setattr(configs, "BS_DIR", str(tmp_path / "bs"))

    configs.copy_model_scripts("m", str(tmp_path / "setup"), cli_edit=False)

    assert os.listdir(tmp_path / "setup") == ["train.py"]


def test_bs_run_refuses_a_stale_run_tail_until_bs_prepare(tmp_path, prepared):
    """A bias edit renames the segmentation that refine reads. Nothing runs until bs prepare
    writes the tail again."""
    segment = tmp_path / "round_01" / "04_segment.toml"
    segment.write_text(segment.read_text().replace("bias = [-0.4", "bias = [-0.3", 1))

    result = CliRunner().invoke(cli, ["run", "round_01"])

    assert result.exit_code == 1
    assert result.stderr == (
        "Error: round_01: 1 run tail is out of date with the keys above it. Nothing ran.\n"
        "  05_refine.toml [run.refine.apical] in_array: "
        "...--b-0.4_-0.4_-0.4_-0.7_-0.7_-0.7_-0.7_-0.7_-0.7... is now "
        "...--b-0.3_-0.4_-0.4_-0.7_-0.7_-0.7_-0.7_-0.7_-0.7...\n"
        "Run: bs prepare round_01\n"
    )
    CliRunner().invoke(prepare, ["round_01"], catch_exceptions=False)
    dry_run = CliRunner().invoke(cli, ["run", "round_01", "--dry-run"]).output
    assert " (shipped)\n" in dry_run, dry_run  # bs prepare downloads it when missing

    train = tmp_path / "round_01" / "01_train_2d_mtlsd.toml"
    train.write_text(train.read_text().replace("= 30001", "= 20001", 1))
    lines = CliRunner().invoke(cli, ["run", "round_01"]).stderr.splitlines()
    assert lines[0].endswith(
        ": 5 run tails are out of date with the keys above them. Nothing ran."
    )
    assert lines[4:] == ["  and 7 more", "Run: bs prepare round_01"]


def test_a_dry_run_prints_what_each_step_reads_and_writes(tmp_path):
    """Each read exists or comes from an earlier step; a missing one stops the run before any
    step."""
    run_prepare(tmp_path, "3M")
    d = f"{tmp_path}/round_01"
    level = f"{d}/apical.zarr/3d_mtlsd/30000"
    seg = step(tmp_path, "04_refine.toml").volumes["apical"]["in_array"].removeprefix(f"{d}/")
    c, rel = f"{tmp_path}/apical.zarr", "apical.zarr/3d_mtlsd/30000"

    result = CliRunner().invoke(cli, ["run", "round_01", "--dry-run"])

    assert result.output == (
        "round_01: 5 steps on 1 volume. Paths are in round_01 unless absolute.\n"
        "01 train.3d_mtlsd\n"
        f"   reads  {c}/raw, {c}/labels, training_mask.zarr/apical\n"
        "   writes 3d_mtlsd\n"
        "02 predict.3d_mtlsd on apical\n"
        f"   reads  {c}/raw, 3d_mtlsd/model_checkpoint_30000.ckpt from 01\n"
        "   writes apical.zarr/3d_mtlsd\n"
        "03 segment on apical\n"
        f"   reads  {rel}/3d_affs from 02\n"
        f"   writes {rel}/fragments_mws, {rel}/luts_mws, {rel}/segmentations_mws\n"
        "04 refine on apical: fill-holes\n"
        f"   reads  {seg} from 03\n"
        f"   writes {seg}_*\n"
        "05 evaluate on apical\n"
        f"   reads  {rel}/segmentations_mws from 03, {c}/labels, {rel}/3d_affs from 02\n"
        "Every input exists or comes from an earlier step. Nothing ran.\n"
    )
    result = CliRunner().invoke(cli, ["run", "round_01", "--from", "03"])
    assert result.stderr == (
        f"Error: 03_segment.toml [segment] apical reads {level}/3d_affs: "
        "nothing there, and no earlier step writes it. Nothing ran.\n"
    )
    assert not os.path.exists(f"{d}/runs")


def test_a_receipt_holds_the_keys_each_step_read_and_dry_runs_clean(tmp_path, prepared):
    refine = step(tmp_path, "05_refine.toml")
    write(refine.volumes["apical"]["in_array"], np.full((8, 64, 64), 7, np.uint64))
    path = tmp_path / "round_01" / "05_refine.toml"
    path.write_text("[refine]\nnum_workers = 1\n" + path.read_text())

    result = CliRunner().invoke(
        cli, ["run", "round_01", "--only", "05"], catch_exceptions=False
    )

    assert result.exit_code == 0, result.output
    (receipt,) = glob.glob(f"{tmp_path}/round_01/runs/*.toml")
    first, status = open(receipt).read().splitlines()[:2]
    assert first.startswith("# bs run round_01 --only 05, ")
    assert re.fullmatch(r"# 05_refine.toml \[refine\]: done, \d+ s", status)
    assert load(receipt) == [step(tmp_path, "05_refine.toml")]
    dry_run = CliRunner().invoke(cli, ["run", receipt, "--dry-run"]).output
    assert dry_run.endswith("comes from an earlier step. Nothing ran.\n"), dry_run


def test_a_glob_that_matches_two_labels_names_the_fix(tmp_path):
    """The shell expands an unquoted * into one argument per match; bs prepare says so and writes nothing."""
    write(tmp_path / "a.zarr" / "raw", np.zeros((8, 16, 16), np.uint8))
    for name in ("seg_rd64_fill_holes", "seg_rd100_fill_holes"):
        write(tmp_path / "a.zarr" / name, np.ones((8, 16, 16), np.uint64))
    matches = sorted(glob.glob(f"{tmp_path}/a.zarr/*_fill_holes"))

    result = CliRunner().invoke(prepare, ["round_02", "--raw", "a.zarr/raw", "--labels", *matches, "--labelled", "densely"])

    assert result.exit_code == 1
    assert f"{matches[1]}: one argument too many" in result.output
    assert "give one array in full" in result.output
    assert not (tmp_path / "round_02").exists()


def test_a_training_mask_that_misses_the_labels_stops_prepare_and_names_the_one_that_fits(tmp_path):
    """As in harris15: three label arrays on different sections, each with its own mask. A mask on
    other sections gives training no labels, and the 2d_mtlsd pipeline crashes in Pad."""
    labels = np.zeros((1, 64, 64), np.uint64)
    labels[0, 10:30, 10:30] = 1
    write(tmp_path / "v.zarr" / "raw", np.full((16, 64, 64), 100, np.uint8))
    write(tmp_path / "v.zarr" / "dense2d_labels", labels, offset=(600, 0, 0))
    write(tmp_path / "v.zarr" / "dense2d_mask", np.ones((1, 64, 64), np.uint8), offset=(600, 0, 0))
    write(tmp_path / "v.zarr" / "sparse_labels_mask", np.ones((1, 64, 64), np.uint8), offset=(200, 0, 0))
    flags = ["--raw", f"{tmp_path}/v.zarr/raw", "--labels", "dense2d_labels"]

    result = CliRunner().invoke(prepare, ["round_01", *flags, "--labelled", "sparse_labels_mask"])
    assert result.exit_code == 1
    assert "sparse_labels_mask does not overlap the labels dense2d_labels" in result.output
    assert "such as dense2d_mask" in result.output
    assert not (tmp_path / "round_01").exists()

    result = CliRunner().invoke(prepare, ["round_01", *flags, "--labelled", "dense2d_mask"], catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_prepare_writes_the_raws_channel_count_as_in_channels(tmp_path):
    """As in prism: a c z y x raw, such as 18 light microscopy channels, trains on all of them."""
    labels = np.zeros((8, 64, 64), np.uint64)
    labels[2, 10:30, 10:30] = 1
    raw = prepare_ds(str(tmp_path / "prism.zarr" / "raw"), shape=(2, *labels.shape), offset=(0, 0, 0),
                     voxel_size=(50, 8, 8), axis_names=["c^", "z", "y", "x"], units=["nm"] * 3, dtype=np.uint8)
    raw[raw.roi] = 100
    write(tmp_path / "prism.zarr" / "labels", labels)
    result = CliRunner().invoke(prepare, [f"{tmp_path}/round_01", "--raw", f"{tmp_path}/prism.zarr/raw", "--labels", "labels",
                                          "--labelled", "sparsely"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    net_config = json.loads((tmp_path / "round_01" / "2d_mtlsd" / "net_config.json").read_text())
    assert net_config["in_channels"] == 2
