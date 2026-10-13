"""`refine morph --op fill_holes --xy` emptied every section it touched.

fastmorph 1.8.0 fills a section by writing its components back one at a time,
and its loop stops one component short. Handed a (1, H, W) array it therefore
dropped whole objects: a section holding one object came back empty.
"""

import numpy as np
from click.testing import CliRunner
from funlib.persistence import open_ds, prepare_ds

from bootstrapper.refine import _fill_holes, morph

HOLE = 40 * 40
OBJECT = 130 * 130 - HOLE


def write_labels(path, arr):
    ds = prepare_ds(
        path,
        shape=arr.shape,
        offset=(0, 0, 0),
        voxel_size=(50, 8, 8),
        axis_names=["z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=np.uint64,
        chunk_shape=(1, 128, 128),
        mode="w",
    )
    ds[ds.roi] = arr


def test_fill_holes_xy_keeps_a_single_object(tmp_path):
    in_array = str(tmp_path / "in.zarr" / "labels")
    out_array = str(tmp_path / "in.zarr" / "labels_fill_holes")
    arr = np.zeros((3, 256, 256), dtype=np.uint64)
    arr[1, 50:180, 50:180] = 7
    arr[1, 80:120, 80:120] = 0
    write_labels(in_array, arr)
    assert int((arr[1] > 0).sum()) == OBJECT

    result = CliRunner().invoke(
        morph,
        ["-i", in_array, "-o", out_array, "--op", "fill_holes", "--xy",
         "-c", "8", "-b", "256", "-w", "1"],
    )

    assert result.exit_code == 0, result.exception
    out = open_ds(out_array)[:]
    assert int((out[1] > 0).sum()) == OBJECT + HOLE
    assert set(np.unique(out[1]).tolist()) == {0, 7}
    assert not out[0].any() and not out[2].any()


def test_fill_holes_xy_keeps_every_object():
    section = np.zeros((120, 200), dtype=np.uint64)
    section[10:50, 10:50] = 3
    section[20:40, 20:40] = 0
    section[10:50, 70:110] = 9
    section[60:110, 10:60] = 41

    filled = _fill_holes(section)

    assert set(np.unique(filled).tolist()) == {0, 3, 9, 41}
    assert int((filled == 3).sum()) == 40 * 40
    assert int((filled == 9).sum()) == 40 * 40
    assert int((filled == 41).sum()) == 50 * 50


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
