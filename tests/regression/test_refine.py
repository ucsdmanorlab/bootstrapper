"""The refine filters judge each object by its whole-volume size, refine mask zeroes labels
outside a mask, and refine morph keeps labels apart.

Fill holes gives every region enclosed by one object that object's id. Enclosed means no
face-connected path through other voxels to the edge of the object's bounding box, in 2D and
3D alike. Background is always filled. An enclosed object goes too only when its region has at
most max_area voxels and holds no excluded id. Closing keeps label contacts, dilate keeps
excluded ids as they are, and the op names are the config file's.
"""

import numpy as np
from click.testing import CliRunner
from funlib.persistence import open_ds, prepare_ds

from bootstrapper.cli import cli
from bootstrapper.config import OPS
from bootstrapper.refine import (
    MORPH_OPS,
    _apply_morph,
    _fill_holes,
    morph,
    outlier_filter,
    size_filter,
)


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
ANY = 10**6  # a max_area above every region here


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
            "fill-holes",
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


def canvas():
    return np.zeros((30, 30), np.uint64)


def test_an_enclosed_object_survives_and_the_gap_around_it_fills():
    a = canvas()
    a[5:25, 5:25] = 7
    a[10:20, 10:20] = 0
    a[13:17, 13:17] = 9
    expected = a.copy()
    expected[10:20, 10:20] = 7
    expected[13:17, 13:17] = 9

    assert np.array_equal(_fill_holes(a), expected)


def test_3d_enclosed_object_survives_unless_max_area_allows():
    volume = np.zeros((7, 20, 20), np.uint64)
    volume[1:6, 3:17, 3:17] = 8
    volume[3, 8:12, 8:12] = 5

    assert np.array_equal(_fill_holes(volume), volume)
    assert np.array_equal(_fill_holes(volume, max_area=ANY), np.where(volume > 0, 8, 0))


def test_enclosed_island_absorbed():
    a = canvas()
    a[5:25, 5:25] = 7
    a[12:16, 12:16] = 9

    out = _fill_holes(a, max_area=ANY)

    assert (out[12:16, 12:16] == 7).all()
    assert (out == 9).sum() == 0


def test_island_touching_two_labels_kept():
    a = canvas()
    a[5:25, 5:15] = 7
    a[5:25, 15:25] = 8
    a[12:16, 13:17] = 9  # straddles 7 and 8

    assert np.array_equal(_fill_holes(a, max_area=ANY), a)


def test_island_touching_background_kept():
    a = canvas()
    a[5:25, 5:25] = 7
    a[12:16, 12:16] = 9
    a[14, 16:26] = 0  # background channel from the island to the exterior

    assert (_fill_holes(a, max_area=ANY) == 9).sum() == 16


def test_diagonal_contact_is_no_opening():
    a = canvas()
    a[5:25, 5:25] = 7
    a[12:16, 12:16] = 9
    a[16:25, 16:25] = 8  # 8 reaches the exterior and touches 9 only at a corner

    out = _fill_holes(a, max_area=ANY)

    assert (out[12:16, 12:16] == 7).all() and (out == 8).sum() == 81


def test_excluded_id_never_absorbed_but_absorbs_others():
    a = canvas()
    a[3:27, 3:27] = 7
    a[8:22, 8:22] = 1  # excluded, enclosed by 7
    a[13:16, 13:16] = 9  # island inside 1

    out = _fill_holes(a, max_area=ANY, exclude=[1])

    assert (out[8:22, 8:22] == 1).all()
    assert (_fill_holes(a, max_area=ANY)[8:22, 8:22] == 7).all()


def test_group_of_touching_islands_absorbed():
    a = canvas()
    a[5:25, 5:25] = 7
    a[10:14, 10:14] = 9
    a[10:14, 14:18] = 10  # 9 and 10 touch each other and 7 only

    assert set(np.unique(_fill_holes(a, max_area=ANY)[5:25, 5:25])) == {7}


def test_max_area_limits_absorption_but_not_filling():
    a = canvas()
    a[2:28, 2:28] = 7
    a[4:16, 4:16] = 0
    a[5:15, 5:15] = 9  # 100 px in a region of 144 px
    a[20:22, 20:22] = 10  # 4 px

    out = _fill_holes(a, max_area=50)

    assert (out == 9).sum() == 100 and (out == 10).sum() == 0
    assert (out[2:28, 2:28] > 0).all()


def test_island_at_array_border_kept():
    a = canvas()
    a[0:20, 5:25] = 7
    a[0:4, 12:16] = 9  # touches the array edge: its outside is unknown

    assert (_fill_holes(a, max_area=ANY) == 9).sum() == 16


def test_gap_between_two_objects_stays():
    a = canvas()
    a[5:25, 5:25] = 7
    a[10:20, 10:20] = 0
    a[14, 20:30] = 8  # 8 reaches the exterior and holds 1 of the gap's 40 border edges

    assert (_fill_holes(a)[10:20, 10:20] == 0).all()


def test_fill_holes_xy_absorbs_per_section(tmp_path):
    in_array = str(tmp_path / "in.zarr" / "labels")
    arr = np.zeros((2, 128, 128), np.uint64)
    arr[:, 20:100, 20:100] = 7
    arr[0, 50:60, 50:60] = 9  # enclosed on section 0
    arr[1, 50:60, 50:128] = 9  # reaches the exterior on section 1
    write_labels(in_array, arr, chunk_shape=(1, 128, 128))
    options = [
        "--op",
        "fill-holes",
        "--xy",
        "--max-area",
        "1000",
        "-c",
        "8",
        "-b",
        "128",
        "-w",
        "1",
    ]

    result = CliRunner().invoke(morph, ["-i", in_array, *options])

    assert result.exit_code == 0, result.output
    out = open_ds(in_array + "_fill_holes")[:]
    assert (out[0] == 9).sum() == 0
    assert np.array_equal(out[1], arr[1])


def neighbours():
    arr = np.zeros((1, 64, 64), np.uint64)
    arr[0, 10:30, 10:25] = 1
    arr[0, 10:30, 28:40] = 2  # 3 px from 1
    arr[0, 40:50, 10:20] = 4
    return arr


def test_every_morph_op_is_a_config_op_name():
    assert set(MORPH_OPS) <= set(OPS)


def test_label_contacts_survive_closing():
    a = np.zeros((30, 30), np.uint64)
    a[5:25, 5:15] = 7
    a[5:25, 15:25] = 8

    assert np.array_equal(_apply_morph(a, "closing", 3), a)


def test_closing_seals_a_neck_that_fill_holes_then_fills():
    a = np.zeros((30, 30), np.uint64)
    a[5:25, 5:25] = 7
    a[10:20, 10:20] = 0  # pocket
    a[14:17, 20:25] = 0  # 3 px wide channel to the exterior

    assert _fill_holes(a)[12, 12] == 0
    assert _fill_holes(_apply_morph(a, "closing", 1))[12, 12] == 0  # 1 px stays open
    assert (_fill_holes(_apply_morph(a, "closing", 2))[10:20, 10:20] == 7).all()


def test_dilate_keeps_excluded_ids_exact(tmp_path):
    arr = neighbours()
    in_array = write_labels(str(tmp_path / "v.zarr" / "labels"), arr)
    options = [
        "--op",
        "dilate",
        "-n",
        "3",
        "--xy",
        "--exclude-ids",
        "1,4",
        "-c",
        "8",
        "-w",
        "1",
    ]

    result = CliRunner().invoke(morph, ["-i", in_array, *options])

    assert result.exit_code == 0, result.output
    out = open_ds(in_array + "_dilate")[:]
    assert np.array_equal(out == 1, arr == 1)
    assert np.array_equal(out == 4, arr == 4)
    assert (out == 2).sum() > (arr == 2).sum()


def test_exclude_ids_from_a_config_file_array(tmp_path):
    arr = neighbours()
    in_array = write_labels(str(tmp_path / "v.zarr" / "labels"), arr)
    config = tmp_path / "refine.toml"
    config.write_text(
        f'[refine]\nin_array = "{in_array}"\n'
        "[refine.dilate]\niterations = 3\nxy = true\nnum_workers = 1\nexclude_ids = [1]\n"
    )

    result = CliRunner().invoke(cli, ["run", str(config)])

    assert result.exit_code == 0, result.output
    out = open_ds(in_array + "_dilate")[:]
    assert np.array_equal(out == 1, arr == 1)
    assert (out == 4).sum() > (arr == 4).sum()


def test_exclude_ids_refuses_a_list_it_cannot_read(tmp_path):
    in_array = write_labels(str(tmp_path / "v.zarr" / "labels"), neighbours())

    result = CliRunner().invoke(
        morph, ["-i", in_array, "--op", "dilate", "--exclude-ids", "1;4"]
    )

    assert result.exit_code != 0
    assert "1;4" in result.output and "1,4,8" in result.output
