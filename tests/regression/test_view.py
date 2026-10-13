"""The viewer binds where it is told, 0.0.0.0 by default (decided on 2026-09-14). It shows a
round's arrays by step, a snapshot, or array paths, each layer typed by one table, and prints
each layer before the URL."""

import contextlib
import importlib
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from click.testing import CliRunner
from funlib.persistence import prepare_ds

from bootstrapper.configs import MODEL_DIR

# bootstrapper/__init__.py rebinds the name "view" to the command
view_module = importlib.import_module("bootstrapper.view")


class ViewerStarted(Exception):
    """Raised in place of building a real neuroglancer viewer."""


@pytest.mark.parametrize(
    "args,bind", [([], "0.0.0.0"), (["--bind", "127.0.0.1"], "127.0.0.1")]
)
def test_view_snapshot_binds_the_address(tmp_path, monkeypatch, args, bind):
    def viewer(*args, **kwargs):
        raise ViewerStarted()

    seen = []
    monkeypatch.setattr(
        view_module.neuroglancer, "set_server_bind_address", seen.append
    )
    monkeypatch.setattr(view_module.neuroglancer, "Viewer", viewer)

    result = CliRunner().invoke(view_module.view, ["--snapshot", str(tmp_path), *args])

    assert isinstance(result.exception, ViewerStarted)
    assert seen == [bind]


@pytest.fixture
def viewer(monkeypatch):
    """The viewer server stubbed: the layers added, by name, and the address bound."""
    state = SimpleNamespace(layers={}, bind=[])

    def append(name, layer, visible=True):
        state.layers[name] = SimpleNamespace(layer=layer, visible=visible)

    txn = SimpleNamespace(layers=SimpleNamespace(append=append), layout=None)
    monkeypatch.setattr(view_module.neuroglancer, "set_server_bind_address", state.bind.append)
    monkeypatch.setattr(
        view_module.neuroglancer,
        "Viewer",
        lambda: SimpleNamespace(txn=lambda: contextlib.nullcontext(txn)),
    )
    return state


def printed(output):
    """(name, path, layer type) of each layer line; the URL line comes last."""
    return [tuple(re.split(r"\s{2,}", line)) for line in output.splitlines()[:-1]]


def array(path, data, axes=("z", "y", "x")):
    channels = ["c^"] if len(axes) < data.ndim else []
    ds = prepare_ds(
        str(path),
        shape=data.shape,
        offset=(0, 0, 0),
        voxel_size=(40, 8, 8),
        axis_names=[*channels, *axes],
        units=["nm"] * 3,
        dtype=data.dtype,
    )
    ds[ds.roi] = data
    return str(path)


@pytest.fixture
def round_dir(tmp_path):
    """A round of three steps on one volume: 2D LSDs, 3D affinities from them, and a
    segmentation of those, each written where its run config says."""
    v = tmp_path / "v.zarr"
    level = v / "3d_affs_from_2d_lsd" / "25000--from--2d_lsd_5000"
    paths = {
        "raw": array(v / "raw", np.zeros((8, 32, 32), np.uint8)),
        "mask": array(v / "mask", np.ones((8, 32, 32), np.uint8)),
        "lsds": array(v / "2d_lsd" / "5000" / "2d_lsds", np.zeros((6, 8, 32, 32), np.uint8)),
        "affs": array(level / "3d_affs", np.zeros((9, 8, 32, 32), np.uint8)),
        "seg": array(level / "segmentations_mws" / "gb1_-0.5--eps0.001--b-0.4", np.zeros((8, 32, 32), np.uint64)),
    }
    path = tmp_path / "round_01"
    path.mkdir()
    (path / "volumes.toml").write_text(
        f'[volumes.v]\nraw = "{paths["raw"]}"\ntarget_mask = "{paths["mask"]}"\n'
    )
    (path / "01_predict_2d_lsd.toml").write_text(
        f'[predict.2d_lsd]\nsetup_dir = "{MODEL_DIR}/2d_lsd"\ncheckpoint = "{MODEL_DIR}/2d_lsd/model_checkpoint_5000"\n'
        f'[predict.2d_lsd.v]\ninput_datasets = ["{paths["raw"]}"]\noutput_datasets_prefix = "{v}/2d_lsd"\n'
    )
    (path / "02_predict_3d_affs_from_2d_lsd.toml").write_text(
        f'[predict.3d_affs_from_2d_lsd]\nsetup_dir = "{MODEL_DIR}/3d_affs_from_2d_lsd"\n'
        f'checkpoint = "{MODEL_DIR}/3d_affs_from_2d_lsd/model_checkpoint_25000"\n'
        f'[predict.3d_affs_from_2d_lsd.v]\ninput_datasets = ["{paths["lsds"]}"]\n'
        f'output_datasets_prefix = "{v}/3d_affs_from_2d_lsd"\nchain_str = "2d_lsd_5000"\n'
    )
    (path / "03_segment.toml").write_text(
        f'[segment]\nmethod = "mws"\n[segment.v]\naffs_dataset = "{paths["affs"]}"\n'
        f'fragments_dataset = "{level}/fragments_mws"\nseg_dataset_prefix = "{level}/segmentations_mws"\n'
    )
    return path, paths


def test_bs_view_of_a_round_shows_each_array_the_round_wrote_with_its_layer(round_dir, viewer):
    path, paths = round_dir

    result = CliRunner().invoke(view_module.view, [str(path)])

    assert result.exit_code == 0, result.output
    assert printed(result.output) == [
        ("raw", paths["raw"], "image"),
        ("target_mask", paths["mask"], "segmentation"),
        ("01 2d_lsds (offset y x, variance y)", paths["lsds"], "rgb image"),
        ("02 3d_affs (z y x)", paths["affs"], "rgb image"),
        ("03 segmentation", paths["seg"], "segmentation"),
    ]
    layers = {name: type(v.layer).__name__ for name, v in viewer.layers.items()}
    assert layers == {
        "raw": "ImageLayer",
        "target_mask": "SegmentationLayer",
        "01 2d_lsds (offset y x, variance y)": "ImageLayer",
        "02 3d_affs (z y x)": "ImageLayer",
        "03 segmentation": "SegmentationLayer",
    }
    assert [name for name, v in viewer.layers.items() if v.visible] == ["raw", "03 segmentation"]
    assert "emitRGB" in viewer.layers["02 3d_affs (z y x)"].layer.shader
    assert "getDataValue(2)" in viewer.layers["01 2d_lsds (offset y x, variance y)"].layer.shader
    assert viewer.bind == ["0.0.0.0"]


@pytest.mark.parametrize("step", ["02", "02_predict_3d_affs_from_2d_lsd.toml"])
def test_bs_view_of_one_step_shows_the_raw_and_what_that_step_wrote(round_dir, viewer, step):
    path, paths = round_dir

    result = CliRunner().invoke(view_module.view, [str(path), step])

    assert result.exit_code == 0, result.output
    assert printed(result.output) == [
        ("raw", paths["raw"], "image"),
        ("02 3d_affs (z y x)", paths["affs"], "rgb image"),
    ]
    assert all(v.visible for v in viewer.layers.values())


def test_bs_view_of_a_step_the_round_lacks_names_the_steps(round_dir, viewer):
    path, _ = round_dir

    result = CliRunner().invoke(view_module.view, [str(path), "07"])

    assert result.exit_code != 0
    assert "07: no such step; steps are 01 predict.2d_lsd, 02 predict.3d_affs_from_2d_lsd, 03 segment" in result.output


def test_bs_view_of_array_paths_shows_each_with_its_layer(affs, viewer):
    result = CliRunner().invoke(view_module.view, [affs, "--bind", "127.0.0.1"])

    assert result.exit_code == 0, result.output
    assert printed(result.output) == [("3d_affs", affs, "rgb image")]
    assert viewer.bind == ["127.0.0.1"]


def test_view_skips_a_group_in_the_snapshot(tmp_path, monkeypatch, viewer):
    path = str(tmp_path / "batch_1.zarr")
    root = zarr.open_group(path, mode="w")
    root.create_group("a_group")
    for name, channels in [("raw", 1), ("pred_affs", 3)]:
        root.create_dataset(name, data=np.zeros((channels, 4, 8, 8), np.uint8))
        root[name].attrs.update(offset=[0, 0, 0], voxel_size=[40, 8, 8])
    # os.listdir order is arbitrary: put the group first
    listdir = os.listdir
    monkeypatch.setattr(
        os, "listdir", lambda p=".": sorted(listdir(p), key=lambda n: n != "a_group")
    )

    result = CliRunner().invoke(view_module.view, ["--snapshot", path])

    assert result.exit_code == 0, result.output
    assert sorted(printed(result.output)) == [
        ("pred_affs", f"{path}/pred_affs", "rgb image"),
        ("raw", f"{path}/raw", "image"),
    ]


def test_bs_view_of_a_scale_pyramid_opens_its_s0(tmp_path, viewer):
    s0 = array(tmp_path / "v.zarr" / "raw" / "s0", np.zeros((8, 32, 32), np.uint8))

    result = CliRunner().invoke(view_module.view, [str(tmp_path / "v.zarr" / "raw")])

    assert result.exit_code == 0, result.output
    assert printed(result.output) == [("raw", s0, "image")]


def test_the_raw_of_a_2d_batch_shows_its_sections_as_rgb(tmp_path, viewer):
    path = str(tmp_path / "batch_1.zarr")
    root = zarr.open_group(path, mode="w")
    for name, shape in [("pred_affs", (2, 6, 1, 4, 4)), ("raw", (2, 3, 8, 8))]:
        root.create_dataset(name, data=np.zeros(shape, np.uint8))
        root[name].attrs.update(offset=[0, 0, 0], voxel_size=[40, 8, 8])

    result = CliRunner().invoke(view_module.view, ["--snapshot", path])

    assert result.exit_code == 0, result.output
    assert ("raw", f"{path}/raw", "rgb image") in printed(result.output)
