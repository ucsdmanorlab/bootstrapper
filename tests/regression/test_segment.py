"""Segmentation: ws and mws are one volara implementation (ticket 09) with a SQLite database
beside the output by default; outputs are named per variant, logged beside their container,
checked against their inputs, and fragments are filtered."""

import glob
import os

import numpy as np
import pytest
import toml
import zarr
from click.testing import CliRunner
from funlib.persistence import prepare_ds
from volara.datasets import Affs, Labels, Raw
from volara.dbs import SQLite

from bootstrapper.blockwise import volara_log_dir
from bootstrapper.cli import cli
from bootstrapper.post.blockwise.extract_frags import ExtractFrags
from bootstrapper.post.blockwise.watershed_frags import WatershedFrags
from bootstrapper.post.naming import build_name, dump_params, fmt, inputs_differ
from bootstrapper.segment import DEFAULTS, get_seg_config, run_segmentation

NEIGHBORHOOD = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]


def test_floats_in_names_keep_twelve_digits():
    assert fmt(0.1234567891234) == "0.123456789123"
    assert fmt(0.5) == "0.5"
    assert build_name({"bias": 0.1000001}) != build_name({"bias": 0.1000002})


def test_volara_logs_go_beside_the_container():
    assert (
        volara_log_dir("/data/x/apical.zarr/segs") == "/data/x/apical_volara_logs/segs"
    )


def test_a_path_with_a_colon_gets_a_hashed_log_dir():
    one = volara_log_dir("/data/a:b/apical.zarr/segs")
    other = volara_log_dir("/data/a:c/apical.zarr/segs")

    assert one.startswith("volara_logs_") and len(one) == len("volara_logs_") + 8
    assert ":" not in one and "=" not in one
    assert one != other


def test_other_inputs_are_named_in_the_warning(tmp_path):
    store = str(tmp_path / "v.zarr" / "frags")
    zarr.open(store, mode="w", shape=(2, 2), dtype="u1")
    run = {
        "affs_dataset": "/a/affs",
        "mask_dataset": None,
        "roi_offset": [0, 0, 0],
        "roi_shape": [1, 2, 3],
    }
    dump_params(store, run)

    warning = inputs_differ(store, {**run, "mask_dataset": "/m/mask"})

    assert inputs_differ(store, dict(run)) is None
    assert (
        "recorded None" in warning
        and "/m/mask" in warning
        and "seg_dataset_prefix" in warning
    )
    assert inputs_differ(str(tmp_path / "nothing"), run) is None


def default_step(tmp_path, affs, method, **extra):
    """The [segment] keys `bs prepare` writes for a default (non-blockwise) run."""
    return {
        "affs_dataset": affs,
        "fragments_dataset": f"{tmp_path}/v.zarr/fragments_{method}",
        "seg_dataset_prefix": f"{tmp_path}/v.zarr/segmentations_{method}",
        "blockwise": False,
        "num_workers": 1,
        "params": {},
        **extra,
    }


@pytest.mark.parametrize("method", ["ws", "mws"])
def test_the_default_database_goes_beside_the_output(tmp_path, affs, capsys, method):
    config = get_seg_config(default_step(tmp_path, affs, method), method)

    db_file = config["db"]["db_file"]
    assert os.path.dirname(db_file) == str(tmp_path / "v.zarr")
    assert os.path.basename(db_file).startswith(f"rag_{method}_") and db_file.endswith(
        ".db"
    )
    assert config["lut_dir"] == str(tmp_path / "v.zarr" / f"luts_{method}")
    assert db_file in capsys.readouterr().out


def test_a_configured_database_is_kept(tmp_path, affs, capsys):
    step = default_step(
        tmp_path,
        affs,
        "ws",
        db={"db_file": "/elsewhere/rag.db"},
        lut_dir="/elsewhere/luts",
    )

    config = get_seg_config(step, "ws")

    assert config["db"] == {"db_file": "/elsewhere/rag.db"}
    assert config["lut_dir"] == "/elsewhere/luts"
    assert capsys.readouterr().out == ""


def test_connected_components_gets_no_database(tmp_path, affs):
    config = get_seg_config(default_step(tmp_path, affs, "cc"), "cc")

    assert "db" not in config
    assert "lut_dir" not in config


@pytest.mark.parametrize("workers,warns", [(18, True), (8, False)])
def test_the_default_database_is_named_per_variant_and_warns_on_many_workers(
    tmp_path, capsys, workers, warns
):
    step = {
        "affs_dataset": "/a/affs",
        "fragments_dataset": f"{tmp_path}/o.zarr/frags",
        "seg_dataset_prefix": f"{tmp_path}/o.zarr/segs",
        "blockwise": True,
        "num_workers": workers,
    }

    db_file = get_seg_config(step, "mws")["db"]["db_file"]

    name = os.path.basename(db_file)
    assert name.startswith("rag_mws_gb") and name.endswith(".db")
    out = capsys.readouterr().out
    warned = f"SQLite locks with {workers} workers" in out and "PostgreSQL" in out
    assert warned == warns


@pytest.mark.parametrize(
    "task_class,affs_data,extra",
    [
        (WatershedFrags, Raw(store="t.zarr/affs"), {}),
        (
            ExtractFrags,
            Affs(store="t.zarr/affs", neighborhood=NEIGHBORHOOD),
            {"bias": [-0.5] * 3},
        ),
    ],
    ids=["ws", "mws"],
)
def test_fragments_drop_the_filtered_fragment(task_class, affs_data, extra):
    """volara's replace_values returns a new array; both tasks dropped it, so filter_fragments did nothing."""
    frags = np.zeros((2, 4, 4), np.uint64)
    frags[:, :2] = 1
    frags[:, 2:] = 2
    affs = np.repeat(np.where(frags == 1, 0.1, 0.9)[None], 3, axis=0)

    class Task(task_class):
        def compute_fragments(self, affs_data):
            return frags.copy()

    task = Task(
        db=SQLite(path="rag.db"),
        affs_data=affs_data,
        frags_data=Labels(store="t.zarr/frags"),
        block_size=(2, 4, 4),
        context=(0, 0, 0),
        filter_fragments=0.5,
        **extra,
    )

    assert list(np.unique(task.get_fragments(affs))) == [0, 2]


def test_float32_three_channel_fragments_equal_float64_all_channel_ones():
    rng = np.random.default_rng(0)
    seg = np.zeros((8, 48, 48), np.uint64)
    for i in range(1, 9):
        z, y, x = rng.integers(0, 4), rng.integers(0, 36), rng.integers(0, 36)
        seg[z : z + 4, y : y + 12, x : x + 12] = i
    # channels 3 to 8 are noise the watershed must ignore
    affs = rng.random((9, 8, 48, 48))
    for k, shift in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        affs[k] = (seg == np.roll(seg, shift, (0, 1, 2))) & (seg > 0)
    ws = {
        k: v
        for k, v in DEFAULTS["ws"].items()
        if k not in ("thresholds", "merge_function")
    }
    task = WatershedFrags.model_construct(
        **ws | {"filter_fragments": 0.0, "remove_debris": 0}
    )

    f64 = task.get_fragments(affs)
    f32 = task.get_fragments(affs[:3].astype(np.float32))

    assert f64.max() > 0
    assert np.array_equal(f64, f32)


def test_a_default_config_segments_with_no_extra_keys(tmp_path, affs):
    run_segmentation(default_step(tmp_path, affs, "ws"), "ws")

    assert len(glob.glob(f"{tmp_path}/v.zarr/rag_ws_*.db")) == 1
    # one segmentation per default threshold
    segs = sorted(glob.glob(f"{tmp_path}/v.zarr/segmentations_ws/*"))
    assert len(segs) == 3

    params = zarr.open(segs[0], mode="r").attrs["bs_params"]
    assert params["blockwise"] is False
    assert params["block_shape"] == [8, 64, 64]  # one block over the whole array
    assert params["filter_fragments"] == 0.1
    assert params["remove_debris"] == 64


def test_a_whole_array_run_is_the_one_block_run(tmp_path, affs):
    """`block_shape = "roi"` and no blockwise give the same one block."""
    step = default_step(
        tmp_path, affs, "ws", blockwise=True, block_shape="roi", num_workers=4
    )

    run_segmentation(step, "ws")

    segs = sorted(glob.glob(f"{tmp_path}/v.zarr/segmentations_ws/*"))
    params = zarr.open(segs[0], mode="r").attrs["bs_params"]
    assert params["blockwise"] is False
    assert params["block_shape"] == [8, 64, 64]
    assert params["context"] == [0, 0, 0]


def mws_step(tmp_path, affs, out="o", **extra):
    step = {
        "affs_dataset": affs,
        "fragments_dataset": f"{tmp_path}/{out}.zarr/frags",
        "seg_dataset_prefix": f"{tmp_path}/{out}.zarr/segs",
        "lut_dir": f"{tmp_path}/{out}.zarr/luts",
        "method": "mws",
        "blockwise": False,
        "roi_offset": [0, 0, 0],
        "roi_shape": [400, 256, 256],
        "params": {
            "aff_neighborhood": NEIGHBORHOOD,
            "bias": [-0.5] * 3,
            "strides": [[1, 1, 1]] * 3,
            "randomized_strides": False,
            "noise_eps": 0.001,
            "filter_fragments": 0.0,
            "remove_debris": 0,
        },
    }
    step.update(extra)
    return step


def write_step(tmp_path, name, step):
    path = tmp_path / f"{name}.toml"
    path.write_text(toml.dumps({"segment": step}))
    return str(path)


def test_a_one_block_region_run_and_a_rerun_with_other_inputs(tmp_path, affs, bs):
    r = bs("segment", write_step(tmp_path, "03_mws_region", mws_step(tmp_path, affs)))

    assert r.returncode == 0, r.stderr
    (frags,) = os.listdir(tmp_path / "o.zarr" / "frags")
    frags = zarr.open(str(tmp_path / "o.zarr" / "frags" / frags), mode="r")
    assert frags.shape == (8, 32, 32)
    assert frags.chunks == (8, 32, 32)  # the region, not the whole array
    assert any(f.startswith("rag_mws_gb") for f in os.listdir(tmp_path / "o.zarr"))
    # one log dir per output prefix: volara drops a task's -meta dir at the start of every run
    assert glob.glob(f"{tmp_path}/o_volara_logs/segs/*-extract-frags-meta")

    mask = prepare_ds(
        f"{tmp_path}/v.zarr/mask",
        shape=(8, 64, 64),
        offset=(0, 0, 0),
        voxel_size=(50, 8, 8),
        axis_names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=np.uint8,
    )
    mask[mask.roi] = 1
    masked = mws_step(tmp_path, affs, mask_dataset=f"{tmp_path}/v.zarr/mask")
    r = bs("segment", write_step(tmp_path, "03_mws_masked", masked), "-mws")

    assert r.returncode == 0, r.stderr
    assert "made from other inputs" in r.stderr and "mask_dataset" in r.stderr


def test_a_neighborhood_longer_than_the_channels_is_refused_before_outputs(
    tmp_path, affs
):
    step = mws_step(tmp_path, affs, out="bad")
    step["params"].update(
        aff_neighborhood=NEIGHBORHOOD * 3, bias=[-0.5] * 9, strides=[[1, 1, 1]] * 9
    )

    result = CliRunner().invoke(
        cli, ["segment", write_step(tmp_path, "03_mws_bad", step)]
    )

    assert result.exit_code != 0
    assert "3 affinity channels" in result.stderr and "9 offsets" in result.stderr
    assert not (tmp_path / "bad.zarr" / "frags").exists()


def test_connected_components_records_its_inputs(tmp_path, affs, bs):
    step = {
        "affs_dataset": affs,
        "fragments_dataset": f"{tmp_path}/c.zarr/frags",
        "seg_dataset_prefix": f"{tmp_path}/c.zarr/segs",
        "blockwise": False,
        "params": {"threshold": 0.5, "remove_debris": 0},
    }

    r = bs("segment", write_step(tmp_path, "03_cc", step), "-cc")

    assert r.returncode == 0, r.stderr
    (seg,) = os.listdir(tmp_path / "c.zarr" / "segs")
    params = zarr.open(str(tmp_path / "c.zarr" / "segs" / seg), mode="r").attrs[
        "bs_params"
    ]
    assert {"affs_dataset", "mask_dataset", "roi_offset", "roi_shape"} <= set(params)
    assert "bootstrapper_version" not in params
