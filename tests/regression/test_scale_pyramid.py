"""The scale pyramid names its datasets right, names the level to give when -i is a pyramid
group, and --labels or --image overrides the guess from the name and dtype."""

import importlib
import os

import numpy as np
import pytest
import zarr
from click.testing import CliRunner
from funlib.persistence import prepare_ds

from bootstrapper.data.scale_pyramid import scale_pyramid


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
    for name in [
        "volume/image/s3",
        "volume/image/s2",
        "volume/image/s1",
        "volume/image/s0",
    ]:
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


def array(path, data):
    ds = prepare_ds(
        path,
        shape=data.shape,
        voxel_size=(50, 8, 8),
        dtype=data.dtype,
        chunk_shape=data.shape,
    )
    ds[ds.roi] = data
    return path


@pytest.mark.parametrize(
    "given,message",
    [
        (
            "pyr.zarr/raw",
            "{d}/pyr.zarr/raw is a group with levels s0, s1; give one level: -i {d}/pyr.zarr/raw/s1",
        ),
        (
            "pyr.zarr",
            "{d}/pyr.zarr is a group, not an array; give the path of an array in it",
        ),
    ],
)
def test_a_group_input_names_what_to_give(tmp_path, given, message):
    zarr.open_group(f"{tmp_path}/pyr.zarr", mode="a").require_group("raw")
    for level in ("s0", "s1"):
        array(f"{tmp_path}/pyr.zarr/raw/{level}", np.zeros((4, 8, 8), np.uint8))

    result = CliRunner().invoke(
        scale_pyramid, ["-i", f"{tmp_path}/{given}", "-s", "1,2,2", "-m", "up"]
    )

    assert result.exit_code == 1
    assert f"Error: {message.format(d=tmp_path)}\n" in result.output


@pytest.mark.parametrize(
    "name,flag,interpolated",
    [("raw_masked", "--image", True), ("raw", "--labels", False)],
)
def test_labels_or_image_overrides_the_guess(tmp_path, name, flag, interpolated):
    data = np.zeros((4, 8, 8), np.uint8)
    data[:, :, 4:] = 200
    path = array(f"{tmp_path}/pyr.zarr/{name}", data)

    result = CliRunner().invoke(
        scale_pyramid, ["-i", path, "-s", "1,2,2", "-m", "up", flag]
    )

    assert result.exit_code == 0, result.output
    values = set(np.unique(zarr.open(f"{path}/s0", "r")[:]).tolist())
    assert (values != {0, 200}) == interpolated
