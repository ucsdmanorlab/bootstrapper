"""Evaluation: error values mean the same thing in every block and every run (ticket 11),
skeleton metrics drop isolated nodes, and every segmentation gets a report."""

import json

import gunpowder as gp
import networkx as nx
import numpy as np
import pytest
from click.testing import CliRunner
from funlib.geometry import Roi
from funlib.persistence import open_ds, prepare_ds

from bootstrapper.cli import cli
from bootstrapper.configs import EVAL_ERROR_THRESHOLDS
from bootstrapper.eval import compute_errors as ce
from bootstrapper.eval.compute_metrics import read_skeletons, seg_report
from bootstrapper.evaluate import get_eval_config, run_pred_evaluation
from bootstrapper.gp import AddAffErrors, AddLSDErrors

NEIGHBORHOOD = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]


def make_volume(
    tmp_path, channels, pred_name, voxel_size=(50, 4, 4), chunk=(4, 16, 16)
):
    """One object filling the labels, and a flat channels-deep prediction beside it."""

    shape = (8, 32, 32)
    labels = prepare_ds(
        f"{tmp_path}/data.zarr/labels",
        shape=shape,
        offset=(0, 0, 0),
        voxel_size=voxel_size,
        axis_names=["z", "y", "x"],
        chunk_shape=chunk,
        dtype=np.uint64,
    )
    labels[labels.roi] = np.ones(shape, dtype=np.uint64)

    pred = prepare_ds(
        f"{tmp_path}/data.zarr/{pred_name}",
        shape=(channels, *shape),
        offset=(0, 0, 0),
        voxel_size=voxel_size,
        axis_names=["c^", "z", "y", "x"],
        chunk_shape=(channels, *chunk),
        dtype=np.uint8,
    )
    pred[pred.roi] = np.full((channels, *shape), 128, dtype=np.uint8)

    return f"{tmp_path}/data.zarr/labels", f"{tmp_path}/data.zarr/{pred_name}"


def aff_node():
    return AddAffErrors(
        gp.ArrayKey("T11_SEG"),
        gp.ArrayKey("T11_SEG_AFFS"),
        gp.ArrayKey("T11_PRED_AFFS"),
        gp.ArrayKey("T11_AFF_MAP"),
        gp.ArrayKey("T11_AFF_MASK"),
        neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
    )


def lsd_node():
    return AddLSDErrors(
        gp.ArrayKey("T11_LSD_SEG"),
        gp.ArrayKey("T11_SEG_LSDS"),
        gp.ArrayKey("T11_PRED_LSDS"),
        gp.ArrayKey("T11_LSD_MAP"),
        gp.ArrayKey("T11_LSD_MASK"),
        sigma=80,
    )


@pytest.mark.parametrize("make_node,channels", [(aff_node, 3), (lsd_node, 10)])
def test_same_difference_gives_same_error_in_every_block(make_node, channels):
    """The divisor is the channel count, not whatever the block happens to hold."""

    node = make_node()
    shared = np.zeros((channels, 1, 2, 2), dtype=np.float32)
    shared[0, 0, 0, 0] = 0.5  # squared difference 0.25 in one channel

    loud = shared.copy()
    loud[:, 0, 1, 1] = 1.0  # a second voxel that differs in every channel

    zeros = np.zeros_like(shared)
    quiet_map = node._create_diff(shared, zeros)
    loud_map = node._create_diff(loud, zeros)

    assert quiet_map[0, 0, 0] == pytest.approx(0.25 / channels)
    assert loud_map[0, 0, 0] == pytest.approx(0.25 / channels)
    assert loud_map[0, 1, 1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "pred_name,channels,key",
    [("3d_affs", 3, "aff_neighborhood"), ("3d_lsds", 10, "lsd_sigma")],
)
def test_a_missing_eval_param_is_refused_before_outputs_exist(
    tmp_path, pred_name, channels, key
):
    seg, pred = make_volume(tmp_path, channels=channels, pred_name=pred_name)

    with pytest.raises(ValueError, match=key):
        ce.compute_errors(
            seg, pred, None, f"{tmp_path}/out.zarr/map", f"{tmp_path}/out.zarr/mask"
        )

    assert not (tmp_path / "out.zarr").exists()


def test_neighborhood_mismatch_is_refused_before_outputs_exist(tmp_path):
    seg, pred = make_volume(tmp_path, channels=6, pred_name="3d_affs")

    with pytest.raises(ValueError, match="6 channels"):
        ce.compute_errors(
            seg,
            pred,
            None,
            f"{tmp_path}/out.zarr/map",
            f"{tmp_path}/out.zarr/mask",
            aff_neighborhood=NEIGHBORHOOD * 3,
        )

    assert not (tmp_path / "out.zarr").exists()


@pytest.mark.parametrize(
    "mode,suffix,volume,given,expected",
    [
        ("gt", True, "a", "report.json", "report_a_gt.json"),
        ("pred", False, None, None, "results_v.json"),
    ],
)
def test_the_results_file_carries_the_volume_and_the_mode_only_when_several_run(
    tmp_path, mode, suffix, volume, given, expected
):
    config_file = f"{tmp_path}/04_eval_v.toml"
    step = {"seg_datasets_prefix": "/x", "gt": {"labels_dataset": "/y"}}

    out_result = given and str(tmp_path / given)
    config = get_eval_config(
        step, config_file, mode, suffix=suffix, volume=volume, out_result=out_result
    )

    assert config["out_result"] == str(tmp_path / expected)


def test_an_odd_lsd_context_gives_an_error_map(tmp_path):
    """sigma 120 at (50, 8, 8) is 7.2 z and 45 xy voxels of context; downsample 2 needs even."""
    seg, pred = make_volume(tmp_path, 10, "3d_lsds", voxel_size=(50, 8, 8))
    out_map = f"{tmp_path}/out.zarr/map"

    ce.compute_errors(
        seg, pred, None, out_map, f"{tmp_path}/out.zarr/mask", lsd_sigma=120
    )

    assert open_ds(out_map)[:].any()


def test_an_roi_smaller_than_one_chunk_gets_an_error_map(tmp_path):
    seg, pred = make_volume(
        tmp_path, 3, "3d_affs", voxel_size=(40, 8, 8), chunk=(8, 32, 32)
    )
    out_map = f"{tmp_path}/e.zarr/map"

    ce.compute_errors(
        seg,
        pred,
        None,
        out_map,
        f"{tmp_path}/e.zarr/mask",
        aff_neighborhood=NEIGHBORHOOD,
        roi_offset=(80, 64, 64),
        roi_shape=(160, 128, 128),
    )

    error_map = open_ds(out_map)
    assert error_map.shape == (4, 16, 16)
    assert error_map[:].all()


@pytest.mark.parametrize(
    "pred_name,channels,params",
    [
        ("3d_affs", 3, {"aff_neighborhood": NEIGHBORHOOD}),
        ("3d_lsds", 10, {"lsd_sigma": 80}),
    ],
)
def test_labels_are_padded_for_any_block_size(tmp_path, pred_name, channels, params):
    seg, pred = make_volume(tmp_path, channels, pred_name, voxel_size=(40, 8, 8))
    out_map = f"{tmp_path}/e.zarr/map"

    ce.compute_errors(seg, pred, None, out_map, f"{tmp_path}/e.zarr/mask", **params)

    assert open_ds(out_map)[:].all()


def test_lsd_error_mask_keeps_a_one_voxel_thin_error():
    error_map = np.zeros((3, 20, 20), np.float32)
    error_map[1, 10, 5:15] = 0.5

    mask = AddLSDErrors._create_mask(None, error_map, (0.1, 1.0))

    assert mask.dtype == np.uint8
    assert int(mask.sum()) == 10 and mask[1, 10, 5:15].all()


@pytest.mark.parametrize(
    "thresholds,kept", [((0.1, 1.0), 0), (EVAL_ERROR_THRESHOLDS["lsds"], 32)]
)
def test_lsd_error_mask_floor(thresholds, kept):
    error_map = np.full((2, 4, 4), 0.05, np.float32)

    assert AddLSDErrors._create_mask(None, error_map, tuple(thresholds)).sum() == kept


def test_error_maps_go_under_out_dir(tmp_path):
    shape = (24, 320, 320)
    rng = np.random.default_rng(1)
    seg = np.zeros(shape, np.uint64)
    for i in range(1, 30):
        z, y, x = rng.integers(0, 20), rng.integers(0, 280), rng.integers(0, 280)
        seg[z : z + 6, y : y + 40, x : x + 40] = i
    affs = np.zeros((3, *shape), np.uint8)
    for k, shift in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        affs[k] = ((seg == np.roll(seg, shift, (0, 1, 2))) & (seg > 0)) * 255
    # the model splits where the segmentation does not
    affs[:, 10:14, 100:200, 100:200] = 0
    for name, data, axes, chunks in [
        ("seg", seg, ["z", "y", "x"], (24, 160, 160)),
        ("3d_affs", affs, ["c^", "z", "y", "x"], (3, 24, 160, 160)),
        ("mask", np.ones(shape, np.uint8), ["z", "y", "x"], (24, 160, 160)),
    ]:
        ds = prepare_ds(
            f"{tmp_path}/v.zarr/{name}",
            shape=data.shape,
            offset=(0, 0, 0),
            voxel_size=(50, 8, 8),
            axis_names=axes,
            units=["nm", "nm", "nm"],
            dtype=data.dtype,
            chunk_shape=chunks,
        )
        ds[ds.roi] = data

    config = {
        "pred": {
            "pred_dataset": f"{tmp_path}/v.zarr/3d_affs",
            "thresholds": [0.1, 1.0],
            "params": {"aff_neighborhood": NEIGHBORHOOD},
        },
        "out_dir": f"{tmp_path}/errs.zarr",
        "mask_dataset": f"{tmp_path}/v.zarr/mask",
    }
    stats = run_pred_evaluation(config, f"{tmp_path}/v.zarr/seg")
    assert stats["mask_ds"] == f"{tmp_path}/v.zarr/mask"
    assert stats["error_map_ds"] == f"{tmp_path}/errs.zarr/seg__vs__3d_affs/error_map"

    error_map = open_ds(stats["error_map_ds"])[:]
    assert error_map[10:14, 100:200, 100:200].mean() > 3 * error_map.mean()


def test_the_step_thresholds_make_the_error_mask(tmp_path):
    seg, pred = make_volume(tmp_path, 3, "3d_affs")
    config = {
        "thresholds": [0.3, 1.0],
        "pred": {"pred_dataset": pred, "params": {"aff_neighborhood": NEIGHBORHOOD}},
    }

    stats = run_pred_evaluation(config, seg)

    # every error is about 0.25: the squared gap from 128/255 to 1 or 0
    assert not open_ds(stats["error_mask_ds"])[:].any()


def test_isolated_skeleton_nodes_are_dropped(tmp_path):
    graph = nx.Graph()
    positions = {
        "1": (5, 5, 5),
        "2": (6, 6, 6),
        "3": (7, 7, 7),
        "4": (8, 8, 8),
        "5": (50, 50, 50),
    }
    for node, (z, y, x) in positions.items():
        graph.add_node(node, position_z=z, position_y=y, position_x=x, id=int(node))
    graph.add_edge("1", "2")  # a skeleton piece inside the ROI
    # 5 is outside the ROI, so the cut leaves 4 alone; 3 is alone from the start
    graph.add_edge("4", "5")
    path = str(tmp_path / "skeletons.graphml")
    nx.write_graphml(graph, path)

    skeletons = read_skeletons(path, Roi((0, 0, 0), (20, 20, 20)))

    assert sorted(skeletons.nodes) == [1, 2]


def write(path, data, voxel_size=(50, 8, 8)):
    ds = prepare_ds(
        path,
        shape=data.shape,
        voxel_size=voxel_size,
        dtype=data.dtype,
        chunk_shape=(2, 10, 10),
    )
    ds[ds.roi] = data
    return path


def test_a_step_without_gt_or_pred_reports_each_segmentation(tmp_path, bs):
    seg = np.zeros((4, 10, 10), np.uint64)
    seg[:, :, :5] = 7  # 200 voxels, over both z slabs
    seg[:, :5, 5:] = 3  # 100
    seg[:2, 5:, 5:] = 9  # 50
    mask = np.zeros(seg.shape, np.uint8)
    mask[:, :, :8] = 1  # 320 voxels; 60 seg voxels lie at x 8 and 9
    write(f"{tmp_path}/o.zarr/segs/a", seg)
    write(f"{tmp_path}/o.zarr/mask", mask)
    config = tmp_path / "04_eval_v.toml"
    config.write_text(
        f'[evaluate]\nseg_datasets_prefix = "{tmp_path}/o.zarr/segs"\n'
        f'mask_dataset = "{tmp_path}/o.zarr/mask"\n'
    )

    r = bs("evaluate", config)

    assert r.returncode == 0, r.stderr
    assert (
        "  3 objects; foreground 0.875 of the ROI, 0.906 of the mask; 60 voxels outside the mask\n"
        "  largest: id 7 = 0.500 of the ROI, id 3 = 0.250, id 9 = 0.125\n"
    ) in r.stdout
    results = json.loads((tmp_path / "results_v.json").read_text())
    assert results[f"{tmp_path}/o.zarr/segs/a"]["report"] == {
        "objects": 3,
        "foreground_of_roi": 0.875,
        "foreground_of_mask": 0.90625,
        "outside_mask": 60,
        "largest": [[7, 0.5], [3, 0.25], [9, 0.125]],
    }


def test_each_volume_is_evaluated_with_its_own_gt_into_its_own_file(tmp_path):
    text = "[volumes.a]\n[volumes.b]\n"
    for v in "ab":
        write(f"{tmp_path}/{v}.zarr/segs/s", np.ones((4, 10, 10), np.uint64))
        text += (
            f'[evaluate.{v}]\nseg_datasets_prefix = "{tmp_path}/{v}.zarr/segs"\n'
            f'[evaluate.{v}.gt]\nlabels_dataset = "{tmp_path}/{v}.zarr/segs/s"\n'
        )
    (tmp_path / "ev.toml").write_text(text)

    result = CliRunner().invoke(cli, ["evaluate", str(tmp_path / "ev.toml")])

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["ev_a.json", "ev_b.json"]
    for v in "ab":
        results = json.loads((tmp_path / f"ev_{v}.json").read_text())
        assert list(results) == [f"{tmp_path}/{v}.zarr/segs/s"]
        assert "voi" in results[f"{tmp_path}/{v}.zarr/segs/s"]["metrics"]


def test_a_mask_on_another_grid_is_refused_by_name(tmp_path):
    seg = write(f"{tmp_path}/o.zarr/seg", np.ones((4, 10, 10), np.uint64))
    mask = write(
        f"{tmp_path}/o.zarr/mask", np.ones((4, 20, 20), np.uint8), voxel_size=(50, 4, 4)
    )

    with pytest.raises(ValueError, match="voxel size") as refused:
        seg_report(seg, mask)

    assert mask in str(refused.value)
