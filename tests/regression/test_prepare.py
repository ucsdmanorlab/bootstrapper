"""bs prepare asks about the data, then writes volumes.toml, numbered step files, a
checkpoint-free setup with no __pycache__, and the training masks in the round, never in the
data container; every file loads, and every param in it says what it does. Each file holds
your keys above a run tail of derived keys, which bs prepare PATH writes again from them.
bs run refuses a stale tail, checks every path before any step, and keeps a receipt."""

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
from bootstrapper.cli import cli
from bootstrapper.config import load, runs
from bootstrapper.params import KEYS, defaults
from bootstrapper.prepare import prepare, write_round

FILES = [
    "01_train_2d_mtlsd.toml",
    "02_predict_2d_mtlsd.toml",
    "03_predict_3d_affs_from_2d_mtlsd.toml",
    "04_segment.toml",
    "05_refine.toml",
    "06_evaluate.toml",
]


def write(path, data, chunk_shape=None, offset=(0, 0, 0)):
    ds = prepare_ds(
        str(path),
        shape=data.shape,
        offset=offset,
        voxel_size=(50, 8, 8),
        axis_names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=data.dtype,
        chunk_shape=chunk_shape,
    )
    ds[ds.roi] = data
    return str(path)


def run_prepare(tmp_path, models="", mask=""):
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
        mask,  # "": no target mask
        f"{tmp_path}/apical.zarr/labels",  # ground truth
        "n",  # no other volume
        models,  # "": the default chain for 2D labels
    ]

    return CliRunner().invoke(
        prepare,
        [f"{tmp_path}/round_01"],
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


def test_your_keys_sit_above_the_run_tail(tmp_path, dialogue):
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


def test_bs_prepare_writes_the_tails_again_and_keeps_your_lines(tmp_path, dialogue):
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
    assert after[predict] == before[predict].replace("_30000.ckpt", "_20000.ckpt")
    third = toml.load(round_dir / "03_predict_3d_affs_from_2d_mtlsd.toml")
    assert "checkpoint" not in third["run"]["predict"]["3d_affs_from_2d_mtlsd"]
    affs = step(tmp_path, "04_segment.toml").volumes["apical"]["affs_dataset"]
    assert affs.endswith("/5000--from--2d_mtlsd_20000/3d_affs")
    mask = open_ds(f"{round_dir}/training_mask.zarr/apical")
    # each square of 20 x 20 grown 1 voxel in xy
    assert mask[mask.roi].sum() == 2 * (20 * 20 + 4 * 20)
    assert net_config.read_text().endswith("}\n\n")
    again = CliRunner().invoke(prepare, ["round_01"])
    assert again.output == "round_01 is current.\n"
    assert {f: f.read_text() for f in round_dir.glob("*.toml")} == after


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


def test_bs_run_refuses_a_stale_run_tail_until_bs_prepare(tmp_path, dialogue):
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
    assert " (shipped)\n" in dry_run, dry_run  # predict downloads it when missing

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
    d, level = f"{tmp_path}/round_01", "3d_mtlsd/30000"
    seg = step(tmp_path, "04_refine.toml").volumes["apical"]["in_array"]
    seg = seg.removeprefix(f"{tmp_path}/apical.zarr/")

    result = CliRunner().invoke(cli, ["run", "round_01", "--dry-run"])

    assert result.output == (
        f"round_01: 5 steps on 1 volume. Paths are in {tmp_path}/apical.zarr unless absolute.\n"
        "01 train.3d_mtlsd\n"
        f"   reads  raw, labels, {d}/training_mask.zarr/apical\n"
        f"   writes {d}/3d_mtlsd\n"
        "02 predict.3d_mtlsd on apical\n"
        f"   reads  raw, {d}/3d_mtlsd/model_checkpoint_30000.ckpt from 01\n"
        "   writes 3d_mtlsd\n"
        "03 segment on apical\n"
        f"   reads  {level}/3d_affs from 02\n"
        f"   writes {level}/fragments_mws, {level}/luts_mws, {level}/segmentations_mws\n"
        "04 refine on apical: fill-holes\n"
        f"   reads  {seg} from 03\n"
        f"   writes {seg}_*\n"
        "05 evaluate on apical\n"
        f"   reads  {level}/segmentations_mws from 03, labels, {level}/3d_affs from 02\n"
        "Every input exists or comes from an earlier step. Nothing ran.\n"
    )
    result = CliRunner().invoke(cli, ["run", "round_01", "--from", "03"])
    assert result.stderr == (
        f"Error: 03_segment.toml [segment] apical reads {tmp_path}/apical.zarr/{level}/3d_affs: "
        "nothing there, and no earlier step writes it. Nothing ran.\n"
    )
    assert not os.path.exists(f"{d}/runs")


def test_a_receipt_holds_the_keys_each_step_read_and_dry_runs_clean(tmp_path, dialogue):
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
