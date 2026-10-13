"""Ticket 02: a mask command must never write over its own input array."""

import click
import numpy as np
import pytest
import zarr
from click.testing import CliRunner

from bootstrapper.data import volumes
from bootstrapper.data.mask import mask
from bootstrapper.data.paths import mask_dataset_path


def make_array(path):
    array = zarr.open(
        str(path), mode="w", shape=(8, 32, 32), chunks=(4, 16, 16), dtype=np.uint8
    )
    array[:] = 200
    array.attrs["offset"] = [0, 0, 0]
    array.attrs["voxel_size"] = [1, 1, 1]
    return str(path)


def unique(path):
    return np.unique(zarr.open(path, mode="r")[:])


def test_derived_path_differs_from_input():
    assert mask_dataset_path("/vol.zarr/labels") == "/vol.zarr/labels_mask"
    assert mask_dataset_path("/vol.zarr/image") == "/vol.zarr/image_mask"


def test_derived_path_of_a_scale_level_masks_into_a_mask_array():
    """A mask is its own array: raw/s0 masks to raw_mask/s0, not raw/s0_mask.

    Otherwise the mask lands inside the pyramid group as a sibling of the
    levels, where the recursive glob that training uses would find it."""
    assert mask_dataset_path("/vol.zarr/raw/s0") == "/vol.zarr/raw_mask/s0"
    assert mask_dataset_path("/vol.zarr/raw/s10") == "/vol.zarr/raw_mask/s10"
    assert mask_dataset_path("/vol.zarr/a/labels/s3") == "/vol.zarr/a/labels_mask/s3"


def test_derived_path_refuses_output_equal_to_input():
    with pytest.raises(click.ClickException):
        mask_dataset_path("/vol.zarr/raw", "//vol.zarr//raw")


def test_mask_command_writes_a_new_dataset(tmp_path):
    in_array = make_array(tmp_path / "vol.zarr" / "objects")

    result = CliRunner().invoke(mask, ["-i", in_array, "-m", "labels"])

    assert result.exit_code == 0, result.output
    assert np.array_equal(unique(in_array), [200])
    assert np.array_equal(unique(in_array + "_mask"), [1])


def test_mask_command_refuses_to_write_over_its_input(tmp_path):
    in_array = make_array(tmp_path / "vol.zarr" / "raw")

    result = CliRunner().invoke(mask, ["-i", in_array, "-o", in_array, "-m", "raw"])

    assert result.exit_code != 0
    assert in_array in result.output
    assert np.array_equal(unique(in_array), [200])


def test_prepare_masks_beside_the_input(tmp_path, monkeypatch):
    container = str(tmp_path / "vol.zarr")
    in_array = make_array(tmp_path / "vol.zarr" / "image")
    calls = []
    monkeypatch.setattr(
        volumes.subprocess, "run", lambda args, **kwargs: calls.append(args)
    )

    # no bounding box, make mask, yes make it
    with CliRunner().isolation(input="n\ny\ny\n"):
        raw_ds, mask_ds, _ = volumes.process_dataset(in_array, container, "raw")

    assert raw_ds == in_array
    assert mask_ds == in_array + "_mask"
    assert calls[0][calls[0].index("-o") + 1] != calls[0][calls[0].index("-i") + 1]
    assert np.array_equal(unique(in_array), [200])
