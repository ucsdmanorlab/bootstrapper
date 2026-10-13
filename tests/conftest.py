import os
import subprocess
import sys

import numpy as np
import pytest
from funlib.persistence import prepare_ds

BS = os.path.join(os.path.dirname(sys.executable), "bs")


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    """Commands write logs into the working directory: keep them in the test's tmp_path."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def bs(tmp_path):
    """Run the bs command in a subprocess; returns the CompletedProcess."""

    def run(*args):
        return subprocess.run(
            [BS, *map(str, args)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=280,
        )

    return run


@pytest.fixture
def affs(tmp_path):
    """Affinities of 11 boxes: 3 channels of 8x64x64 voxels in v.zarr/3d_affs."""
    rng = np.random.default_rng(0)
    seg = np.zeros((8, 64, 64), np.uint64)
    for i in range(1, 12):
        z, y, x = rng.integers(0, 4), rng.integers(0, 50), rng.integers(0, 50)
        seg[z : z + 4, y : y + 14, x : x + 14] = i
    data = np.zeros((3, *seg.shape), np.uint8)
    for k, shift in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        data[k] = ((seg == np.roll(seg, shift, (0, 1, 2))) & (seg > 0)) * 255
    path = str(tmp_path / "v.zarr" / "3d_affs")
    ds = prepare_ds(
        path,
        shape=data.shape,
        offset=(0, 0, 0),
        voxel_size=(50, 8, 8),
        axis_names=["c^", "z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=np.uint8,
        chunk_shape=(3, 8, 32, 32),
    )
    ds[ds.roi] = data
    return path
