"""bs utils mask writes a new array beside its input (ticket 02): a closed image mask or an
object mask, never over the input."""

import numpy as np
import zarr
from click.testing import CliRunner

from bootstrapper.data.mask import mask
from bootstrapper.data.paths import mask_dataset_path


def make_array(path, data):
    array = zarr.open(
        str(path), mode="w", shape=data.shape, chunks=(4, 16, 16), dtype=np.uint8
    )
    array[:] = data
    array.attrs["offset"] = [0, 0, 0]
    array.attrs["voxel_size"] = [1, 1, 1]
    return str(path)


def unique(path):
    return np.unique(zarr.open(path, mode="r")[:])


def test_the_mask_path_is_its_own_array():
    """raw/s0 masks to raw_mask/s0, not raw/s0_mask: inside the pyramid group the mask
    is a sibling of the levels, where the recursive glob that training uses finds it."""
    assert mask_dataset_path("/vol.zarr/labels") == "/vol.zarr/labels_mask"
    assert mask_dataset_path("/vol.zarr/raw/s10") == "/vol.zarr/raw_mask/s10"
    assert mask_dataset_path("/vol.zarr/a/labels/s3") == "/vol.zarr/a/labels_mask/s3"


def test_mask_command_writes_a_new_dataset(tmp_path):
    in_array = make_array(tmp_path / "vol.zarr" / "objects", np.full((8, 32, 32), 200))

    result = CliRunner().invoke(mask, ["-i", in_array, "-m", "labels"])

    assert result.exit_code == 0, result.output
    assert np.array_equal(unique(in_array), [200])
    assert np.array_equal(unique(in_array + "_mask"), [1])


def test_mask_command_refuses_to_write_over_its_input(tmp_path):
    in_array = make_array(tmp_path / "vol.zarr" / "raw", np.full((8, 32, 32), 200))
    same = in_array.replace("/vol.zarr/", "//vol.zarr//")

    result = CliRunner().invoke(mask, ["-i", in_array, "-o", same, "-m", "raw"])

    assert result.exit_code != 0
    assert in_array in result.output
    assert np.array_equal(unique(in_array), [200])


def test_an_image_mask_closes_dark_spots_in_the_tissue(tmp_path):
    tissue = np.zeros((8, 32, 32), np.uint8)
    tissue[:, 4:28, 4:28] = 1
    raw = tissue * 200
    raw[:, 14:17, 14:17] = 0
    in_array = make_array(tmp_path / "vol.zarr" / "raw", raw)
    out_array = str(tmp_path / "out.zarr" / "tissue")

    result = CliRunner().invoke(mask, ["-i", in_array, "-o", out_array, "-m", "raw"])

    assert result.exit_code == 0, result.output
    assert np.array_equal(zarr.open(out_array, mode="r")[:], tissue)
