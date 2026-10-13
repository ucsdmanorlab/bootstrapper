"""The refine filters judge each object by its whole-volume size, refine mask zeroes labels
outside a mask, and fill holes gives every enclosed region the id of the object around it."""

import numpy as np
from click.testing import CliRunner
from funlib.persistence import open_ds, prepare_ds

from bootstrapper.refine import _fill_holes, morph, outlier_filter, size_filter


def write_labels(path, arr, voxel_size=(50, 8, 8), chunk_shape=None):
    ds = prepare_ds(
        path,
        shape=arr.shape,
        offset=(0, 0, 0),
        voxel_size=voxel_size,
        axis_names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=arr.dtype,
        chunk_shape=chunk_shape,
    )
    ds[ds.roi] = arr
    return path


def ids(path):
    return set(np.unique(open_ds(path)[:]).tolist()) - {0}


def test_outlier_filter_cuts_both_tails(tmp_path):
    arr = np.zeros((1, 40, 400), np.uint64)
    for i in range(10):
        arr[0, 0:10, 20 * i : 20 * i + 10] = i + 1  # ten objects of 100 voxels
    arr[0, 20:25, 0:10] = 11  # 50 voxels
    arr[0, 20:30, 20:35] = 12  # 150 voxels
    in_array = write_labels(str(tmp_path / "v.zarr" / "labels"), arr)

    result = CliRunner().invoke(outlier_filter, ["-i", in_array, "-n", "2", "-w", "1"])

    assert result.exit_code == 0, result.output
    out = in_array + "_outlier_filtered"
    assert ids(out) == set(range(1, 11))
    assert np.array_equal(open_ds(out)[:], np.where(arr > 10, 0, arr))


def test_size_filter_keeps_the_size_range(tmp_path):
    arr = np.zeros((4, 64, 64), np.uint64)
    arr[0, 0:3, 0:3] = 1  # 9 voxels
    arr[0:4, 10:20, 10:20] = 2  # 400 voxels
    arr[0:4, 30:60, 30:60] = 3  # 3600 voxels, over two xy chunks
    in_array = write_labels(
        str(tmp_path / "v.zarr" / "labels"), arr, chunk_shape=(4, 32, 32)
    )
    out = str(tmp_path / "v.zarr" / "sized")

    options = ["--min-size", "10", "--max-size", "1000", "-w", "1"]
    result = CliRunner().invoke(size_filter, ["-i", in_array, "-o", out, *options])

    assert result.exit_code == 0, result.output
    assert ids(out) == {2}


def test_refine_mask_zeroes_labels_outside_the_mask(tmp_path, bs):
    shape = (8, 300, 300)
    labels = np.random.default_rng(0).integers(0, 5, shape).astype(np.uint64) * 1000003
    in_array = write_labels(
        str(tmp_path / "v.zarr" / "labels"), labels, chunk_shape=(8, 100, 100)
    )
    mask = np.ones((8, 200, 300), np.uint8)  # covers y 0:200 only
    mask[:, 50:100, 50:100] = 0
    mask_array = write_labels(
        str(tmp_path / "v.zarr" / "mask"), mask, chunk_shape=(8, 100, 100)
    )

    r = bs("refine", "mask", "-i", in_array, "-m", mask_array, "-w", "2")

    assert r.returncode == 0, r.stderr
    want = labels.copy()
    want[:, 200:, :] = 0  # beyond the mask's extent counts as outside
    want[:, 50:100, 50:100] = 0
    assert np.array_equal(open_ds(in_array + "_masked")[:], want)


def test_refine_mask_refuses_another_voxel_size(tmp_path, bs):
    in_array = write_labels(
        str(tmp_path / "v.zarr" / "labels"), np.ones((8, 32, 32), np.uint64)
    )
    mask_array = write_labels(
        str(tmp_path / "v.zarr" / "mask_s1"),
        np.ones((8, 16, 16), np.uint8),
        voxel_size=(50, 16, 16),
    )

    out = tmp_path / "v.zarr" / "x"
    r = bs("refine", "mask", "-i", in_array, "-m", mask_array, "-o", out)

    assert r.returncode == 1
    assert "voxel size" in r.stderr and mask_array in r.stderr
    assert "Traceback" not in r.stderr
    assert not out.exists()


HOLE = 40 * 40
OBJECT = 130 * 130 - HOLE


def test_fill_holes_xy_keeps_a_single_object(tmp_path):
    """fastmorph 1.8.0 dropped the last component of a section: one object came back empty."""
    in_array = str(tmp_path / "in.zarr" / "labels")
    out_array = str(tmp_path / "in.zarr" / "labels_fill_holes")
    arr = np.zeros((3, 256, 256), dtype=np.uint64)
    arr[1, 50:180, 50:180] = 7
    arr[1, 80:120, 80:120] = 0
    write_labels(in_array, arr, chunk_shape=(1, 128, 128))
    assert int((arr[1] > 0).sum()) == OBJECT

    result = CliRunner().invoke(
        morph,
        [
            "-i",
            in_array,
            "-o",
            out_array,
            "--op",
            "fill_holes",
            "--xy",
            "-c",
            "8",
            "-b",
            "256",
            "-w",
            "1",
        ],
    )

    assert result.exit_code == 0, result.exception
    out = open_ds(out_array)[:]
    assert int((out[1] > 0).sum()) == OBJECT + HOLE
    assert set(np.unique(out[1]).tolist()) == {0, 7}
    assert not out[0].any() and not out[2].any()


def test_fill_holes_xy_leaves_an_open_concavity():
    section = np.zeros((60, 60), dtype=np.uint64)
    section[10:50, 10:50] = 6
    section[20:40, 20:60] = 0

    assert np.array_equal(_fill_holes(section), section)


def test_fill_holes_3d_fills_an_enclosed_hole():
    volume = np.zeros((5, 60, 60), dtype=np.uint64)
    volume[1:4, 10:50, 10:50] = 8
    volume[2:3, 20:40, 20:40] = 0

    filled = _fill_holes(volume)

    assert set(np.unique(filled).tolist()) == {0, 8}
    assert int((filled == 8).sum()) == 3 * 40 * 40
