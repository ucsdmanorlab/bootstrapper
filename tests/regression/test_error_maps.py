"""Ticket 11: error values mean the same thing in every block and every run."""

import json
import os

import numpy as np
import pytest
import toml
from funlib.persistence import prepare_ds

import gunpowder as gp

import bootstrapper
from bootstrapper.eval import compute_errors as ce
from bootstrapper.evaluate import get_eval_config, run_pred_evaluation
from bootstrapper.gp import AddAffErrors, AddLSDErrors


def trained_with(model, output_name, key):
    """What the shipped model was trained with."""

    net_config_file = os.path.join(
        os.path.dirname(bootstrapper.__file__), "models", model, "net_config.json"
    )
    with open(net_config_file, "r") as f:
        return json.load(f)["outputs"][output_name][key]


def make_volume(tmp_path, channels, pred_name, voxel_size=(50, 4, 4)):
    """A labels array and a channels-deep prediction beside it."""

    shape = (8, 32, 32)
    labels = prepare_ds(
        f"{tmp_path}/data.zarr/labels",
        shape=shape,
        offset=(0, 0, 0),
        voxel_size=voxel_size,
        axis_names=["z", "y", "x"],
        dtype=np.uint64,
    )
    labels[labels.roi] = np.ones(shape, dtype=np.uint64)

    pred = prepare_ds(
        f"{tmp_path}/data.zarr/{pred_name}",
        shape=(channels, *shape),
        offset=(0, 0, 0),
        voxel_size=voxel_size,
        axis_names=["c^", "z", "y", "x"],
        chunk_shape=(channels, 4, 16, 16),
        dtype=np.uint8,
    )
    pred[pred.roi] = np.zeros((channels, *shape), dtype=np.uint8)

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


def test_neighborhood_defaults_to_one_offset_per_channel(tmp_path, monkeypatch):
    seg, pred = make_volume(tmp_path, channels=9, pred_name="3d_affs")

    captured = {}

    class Stop(Exception):
        pass

    def stub(*args, **kwargs):
        captured["neighborhood"] = args[5]
        raise Stop()

    monkeypatch.setattr(ce, "AddAffErrors", stub)

    with pytest.raises(Stop):
        ce.compute_errors(
            seg, pred, None, f"{tmp_path}/out.zarr/map", f"{tmp_path}/out.zarr/mask"
        )

    model = trained_with("3d_affs", "3d_affs", "neighborhood")
    assert captured["neighborhood"] == model[:9]
    assert len(captured["neighborhood"]) == 9


def test_neighborhood_mismatch_is_refused_before_outputs_exist(tmp_path):
    seg, pred = make_volume(tmp_path, channels=6, pred_name="3d_affs")

    with pytest.raises(ValueError, match="6 channels"):
        ce.compute_errors(
            seg,
            pred,
            None,
            f"{tmp_path}/out.zarr/map",
            f"{tmp_path}/out.zarr/mask",
            aff_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]] * 3,
        )

    assert not (tmp_path / "out.zarr").exists()


def test_lsd_sigma_comes_from_the_model(tmp_path, monkeypatch):
    seg, pred = make_volume(tmp_path, channels=10, pred_name="3d_lsds")

    captured = {}

    class Stop(Exception):
        pass

    def stub(*args, **kwargs):
        captured["sigma"] = kwargs["sigma"]
        raise Stop()

    monkeypatch.setattr(ce, "AddLSDErrors", stub)

    with pytest.raises(Stop):
        ce.compute_errors(
            seg, pred, None, f"{tmp_path}/out.zarr/map", f"{tmp_path}/out.zarr/mask"
        )

    assert captured["sigma"] == trained_with("3d_lsd", "3d_lsds", "sigma")
    assert captured["sigma"] != 4 * 10  # the voxel size fallback on a 50/4/4 volume


def test_each_mode_writes_its_own_results_file(tmp_path):
    config_file = f"{tmp_path}/04_eval_volume_1.toml"
    with open(config_file, "w") as f:
        toml.dump({"seg_datasets": [f"{tmp_path}/data.zarr/seg"]}, f)

    given = f"{tmp_path}/results.json"
    gt = get_eval_config(config_file, "gt", out_result=given)["out_result"]
    pred = get_eval_config(config_file, "pred", out_result=given)["out_result"]

    assert gt != pred


def test_results_record_the_input_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "compute_errors", lambda *args, **kwargs: None)
    monkeypatch.setattr(ce, "compute_stats", lambda array: {})
    monkeypatch.setattr(
        "bootstrapper.evaluate.open_ds", lambda *args, **kwargs: np.zeros((1, 1, 1))
    )

    config = {
        "mask_dataset": f"{tmp_path}/data.zarr/mask",
        "pred": {"pred_dataset": f"{tmp_path}/data.zarr/3d_affs"},
    }
    stats = run_pred_evaluation(config, f"{tmp_path}/data.zarr/seg")

    assert stats["mask_ds"] == config["mask_dataset"]
