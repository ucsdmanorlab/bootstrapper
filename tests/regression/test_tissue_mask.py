"""bs utils tissue-mask: folds and black sections stay in the mask, holes close, wide bays stay out."""

import numpy as np
import zarr
from click.testing import CliRunner
from funlib.persistence import prepare_ds
from scipy import ndimage

from bootstrapper.data.tissue_mask import tissue_mask


def make_raw(path):
    raw = np.zeros((12, 96, 96), np.uint8)
    raw[:, 8:88, 8:88] = 100
    raw[:, 16:26, 40:50] = 0  # hole
    raw[:, 16:20, 80:88] = 0  # bay 4 px wide, narrower than the disk
    raw[:, 70:84, 8:30] = 0  # bay 14 px wide, open to the image border
    raw[6, 34:44, 8:88] = 0  # fold on one section
    raw[2:6, 52:62, 8:88] = 0  # fold on four sections, across the block edge at z 4
    raw[8, :, 8:12] = 0  # image edge shifted on one section
    raw[9] = 0  # black section
    ds = prepare_ds(
        str(path),
        shape=raw.shape,
        voxel_size=(50, 8, 8),
        dtype=np.uint8,
        chunk_shape=(4, 48, 48),
    )
    ds[ds.roi] = raw
    return str(path), raw


def test_tissue_mask_fills_folds_black_sections_and_holes(tmp_path):
    path, raw = make_raw(tmp_path / "vol.zarr" / "raw")
    out = str(tmp_path / "vol.zarr" / "tissue")

    result = CliRunner().invoke(tissue_mask, [path, out, "--radius", "3", "-w", "2"])

    assert result.exit_code == 0, result.output
    mask = zarr.open(out, "r")[:] > 0
    assert (np.delete(mask, 8, 0) == mask[0]).all(), (
        "folds and the black section are filled"
    )
    assert not mask[8, :, 8:12].any(), "z filling stays inside the section's own hull"
    assert all((ndimage.binary_fill_holes(s) == s).all() for s in mask)
    assert mask[raw > 0].all()
    assert mask[0, 16:26, 40:50].all() and mask[0, 16:20, 80:86].all(), (
        "hole and narrow bay close"
    )
    assert not mask[0, 77, 8:30].any(), "a bay wider than the disk stays out"
    attrs = zarr.open(out, "r").attrs
    assert attrs["filled_sections"] == [9] and attrs["voxel_size"] == [50, 8, 8]
    assert "x^2 + y^2 <= 3^2" in attrs["rule"]


def test_tissue_mask_refuses_an_existing_out(tmp_path):
    path, raw = make_raw(tmp_path / "vol.zarr" / "raw")

    result = CliRunner().invoke(tissue_mask, [path, path])

    assert result.exit_code != 0
    assert path in result.output
    assert (zarr.open(path, "r")[:] == raw).all()
