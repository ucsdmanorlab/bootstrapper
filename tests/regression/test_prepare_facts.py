"""bs prepare reads the facts of a PRISM-like volume right: labels on sections that fall between
the sampled ones."""

import numpy as np
import pytest
import toml
from click.testing import CliRunner
from funlib.persistence import prepare_ds

from bootstrapper.prepare import prepare


def write(path, data):
    ds = prepare_ds(str(path), shape=data.shape, offset=(0, 0, 0), voxel_size=(400, 168, 168),
                    axis_names=["z", "y", "x"], units=["nm"] * 3, dtype=data.dtype)
    ds[ds.roi] = data
    return str(path)


@pytest.mark.parametrize("sections, flat", [((10, 31, 52), True), ((10, 11), False), ((0, 30, 31), False)])
def test_labels_between_the_sampled_sections_are_found(tmp_path, sections, flat):
    """200 sections give samples 0, 12, 24, ...: labels on 10 and 11 are 3D, on 10, 31, 52 2D.
    On 0, 30, 31 the sample finds only section 0, and the pair 30, 31 makes the labels 3D."""
    labels = np.zeros((200, 16, 16), np.uint64)
    labels[list(sections), 4:12, 4:12] = 1
    write(tmp_path / "v.zarr" / "raw", np.zeros(labels.shape, np.uint8))
    write(tmp_path / "v.zarr" / "sparse_labels", labels)
    result = CliRunner().invoke(prepare, [f"{tmp_path}/round_01", "--raw", f"{tmp_path}/v.zarr/raw", "--labels",
                                          "sparse_labels", "--labelled", "densely"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert f"labels on {len(sections)} of 200" in result.output
    volume = toml.load(tmp_path / "round_01" / "volumes.toml")["volumes"]["v-zarr"]
    assert volume["2d_labels"] is flat
