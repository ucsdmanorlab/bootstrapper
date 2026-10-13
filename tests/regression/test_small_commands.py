"""The viewer stays on this machine, and the scale pyramid names its datasets right."""

import importlib
import os

import numpy as np
import pytest
import zarr
from click.testing import CliRunner

from bootstrapper.data.scale_pyramid import scale_pyramid

# bootstrapper/__init__.py rebinds the name "view" to the command
view_module = importlib.import_module("bootstrapper.view")


class ViewerStarted(Exception):
    """Raised in place of building a real neuroglancer viewer."""


def test_view_binds_to_this_machine_by_default(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.zarr"
    snapshot.mkdir()
    seen = {}
    monkeypatch.setattr(
        view_module, "view_snapshot", lambda path, bind: seen.update(bind=bind)
    )

    result = CliRunner().invoke(view_module.view, ["--snapshot", str(snapshot)])

    assert result.exit_code == 0, result.exception
    assert seen["bind"] == "127.0.0.1"


def test_view_bind_option_opens_the_viewer_wider(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.zarr"
    snapshot.mkdir()
    seen = {}
    monkeypatch.setattr(
        view_module, "view_snapshot", lambda path, bind: seen.update(bind=bind)
    )

    result = CliRunner().invoke(
        view_module.view, ["--snapshot", str(snapshot), "--bind", "0.0.0.0"]
    )

    assert result.exit_code == 0, result.exception
    assert seen["bind"] == "0.0.0.0"


def test_view_snapshot_binds_to_the_address_it_is_given(tmp_path, monkeypatch):
    def viewer(*args, **kwargs):
        raise ViewerStarted()

    seen = []
    monkeypatch.setattr(
        view_module.neuroglancer, "set_server_bind_address", seen.append
    )
    monkeypatch.setattr(view_module.neuroglancer, "Viewer", viewer)

    with pytest.raises(ViewerStarted):
        view_module.view_snapshot(str(tmp_path / "snapshot.zarr"), "127.0.0.1")

    assert seen == ["127.0.0.1"]


def test_view_datasets_binds_the_neuroglancer_process(monkeypatch):
    seen = []
    monkeypatch.setattr(view_module.subprocess, "run", lambda args, **kw: seen.append(args))

    result = CliRunner().invoke(view_module.view, ["some.zarr/raw"])

    assert result.exit_code == 0, result.exception
    assert seen == [["neuroglancer", "-b", "127.0.0.1", "-d", "some.zarr/raw"]]


@pytest.fixture
def container(tmp_path, monkeypatch):
    """A zarr container holding one array, with the blockwise scaling stubbed out."""
    monkeypatch.setattr(
        importlib.import_module("bootstrapper.data.scale_pyramid"),
        "scale_array",
        lambda *args, **kwargs: None,
    )

    def make(ds_name):
        path = str(tmp_path / "test.zarr")
        zarr.open_group(path, mode="a")  # the container root, not a side effect
        array = zarr.open(
            os.path.join(path, ds_name), mode="w", shape=(16, 16, 16), dtype=np.uint8
        )
        array[:] = 1
        array.attrs["voxel_size"] = (8, 8, 8)
        array.attrs["offset"] = (0, 0, 0)
        return path

    return make


def run_scale_pyramid(path, ds_name, mode, scales):
    args = ["-i", os.path.join(path, ds_name)]
    for scale in scales:
        args += ["-s", ",".join(str(s) for s in scale)]
    args += ["-c", "8,8,8", "-m", mode]
    return CliRunner().invoke(scale_pyramid, args)


def test_upscale_renumbers_the_input_dataset(container):
    path = container("volume/image/s2")

    result = run_scale_pyramid(path, "volume/image/s2", "up", [(2, 2, 2)] * 3)

    assert result.exit_code == 0, result.exception
    group = zarr.open(path, mode="r")
    # the input moves to s3, so the three new scales are s2, s1, s0
    for name in ["volume/image/s3", "volume/image/s2", "volume/image/s1", "volume/image/s0"]:
        assert name in group, f"{name} is missing"
    assert "volume/s1" not in group
    assert "volume/image/s-1" not in group


def test_upscale_handles_multi_digit_scale_numbers(container):
    path = container("image")

    result = run_scale_pyramid(path, "image", "up", [(1, 1, 1)] * 12)

    assert result.exit_code == 0, result.exception
    group = zarr.open(path, mode="r")
    # the input moves to s12, so the twelve new scales are s11 down to s0
    for scale in range(13):
        assert f"image/s{scale}" in group, f"image/s{scale} is missing"
    assert "image/s-1" not in group
