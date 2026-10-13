import numpy as np
import tifffile
from click.testing import CliRunner
from funlib.persistence import open_ds, prepare_ds

from bootstrapper.data.bbox import bbox
from bootstrapper.data.convert import convert


def write_labels(path):
    ds = prepare_ds(
        path,
        shape=(10, 20, 20),
        offset=(0, 0, 0),
        voxel_size=(40, 8, 8),
        axis_names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=np.uint64,
        mode="w",
    )
    arr = np.zeros((10, 20, 20), dtype=np.uint64)
    arr[2:5, 3:9, 4:11] = 3
    ds[ds.roi] = arr


def test_bbox_crops_labels(tmp_path):
    in_array = str(tmp_path / "in.zarr" / "labels")
    out_array = str(tmp_path / "out.zarr" / "labels")
    write_labels(in_array)

    result = CliRunner().invoke(bbox, ["-i", in_array, "-o", out_array])

    assert result.exit_code == 0, result.exception
    out_ds = open_ds(out_array)
    assert out_ds.shape == (3, 6, 7)
    assert tuple(out_ds.offset) == (80, 24, 32)


def test_convert_crops_labels(tmp_path):
    in_path = str(tmp_path / "labels.tif")
    out_array = str(tmp_path / "out.zarr" / "labels")
    arr = np.zeros((10, 20, 20), dtype=np.uint64)
    arr[2:5, 3:9, 4:11] = 3
    tifffile.imwrite(in_path, arr)

    args = ["-i", in_path, "-o", out_array, "-d", "uint64"]
    args += ["-vs", "40", "8", "8", "--crop"]
    result = CliRunner().invoke(convert, args)

    assert result.exit_code == 0, result.exception
    out_ds = open_ds(out_array)
    assert out_ds.shape == (3, 6, 7)
    assert tuple(out_ds.offset) == (80, 24, 32)
