"""Ticket 09: watershed and mutex watershed have one implementation.

The in-memory paths are deleted, so every ws and mws run is a volara task and
a whole-array run is the one-block case. That makes a graph database
mandatory, so a config without one gets a SQLite database beside its output.
Connected components keeps its whole-array path and still needs no database
(ADR-0012).
"""

import glob
import os

import numpy as np
import toml
import zarr
from funlib.persistence import prepare_ds

from bootstrapper.post import watershed, watershed_mutex
from bootstrapper.segment import get_seg_config, run_segmentation


def make_affs(container):
    """Affinities with three low-affinity planes: 8 blobs, 20x64x64 voxels."""
    shape = (20, 64, 64)
    data = np.full((3,) + shape, 250, dtype=np.uint8)
    data[:, 9:11, :, :] = 5
    data[:, :, 20:22, :] = 5
    data[:, :, :, 30:32] = 5

    array = prepare_ds(
        os.path.join(container, "3d_affs"),
        shape=(3,) + shape,
        offset=(0, 0, 0),
        voxel_size=(50, 8, 8),
        axis_names=["c^", "z", "y", "x"],
        units=["nm", "nm", "nm"],
        dtype=np.uint8,
    )
    array[:] = data
    return os.path.join(container, "3d_affs")


def write_config(tmp_path, method, extra=None):
    """The keys `bs prepare` writes for a default (non-blockwise) run."""
    container = str(tmp_path / "vol.zarr")
    config = {
        "affs_dataset": make_affs(container),
        "fragments_dataset": os.path.join(container, f"fragments_{method}"),
        "seg_dataset_prefix": os.path.join(container, f"segmentations_{method}"),
        "blockwise": False,
        "num_workers": 1,
        f"{method}_params": {},
    }
    config.update(extra or {})
    path = str(tmp_path / f"seg_{method}.toml")
    with open(path, "w") as f:
        toml.dump(config, f)
    return path


def test_the_in_memory_paths_are_gone():
    assert not hasattr(watershed, "simple_watershed")
    assert not hasattr(watershed_mutex, "simple_mutex")


def test_watershed_config_defaults_the_database_beside_the_output(tmp_path, capsys):
    config_file = write_config(tmp_path, "ws")

    config = get_seg_config(config_file, "ws")

    db_file = str(tmp_path / "vol.zarr" / "rag_ws.db")
    assert config["db"] == {"db_file": db_file}
    assert config["lut_dir"] == str(tmp_path / "vol.zarr" / "luts_ws")
    assert db_file in capsys.readouterr().out


def test_mutex_config_defaults_the_database_beside_the_output(tmp_path):
    config_file = write_config(tmp_path, "mws")

    config = get_seg_config(config_file, "mws")

    assert config["db"] == {"db_file": str(tmp_path / "vol.zarr" / "rag_mws.db")}
    assert config["lut_dir"] == str(tmp_path / "vol.zarr" / "luts_mws")


def test_a_configured_database_is_kept(tmp_path, capsys):
    config_file = write_config(
        tmp_path,
        "ws",
        {"db": {"db_file": "/elsewhere/rag.db"}, "lut_dir": "/elsewhere/luts"},
    )

    config = get_seg_config(config_file, "ws")

    assert config["db"] == {"db_file": "/elsewhere/rag.db"}
    assert config["lut_dir"] == "/elsewhere/luts"
    assert capsys.readouterr().out == ""


def test_connected_components_gets_no_database(tmp_path):
    config_file = write_config(tmp_path, "cc")

    config = get_seg_config(config_file, "cc")

    assert "db" not in config
    assert "lut_dir" not in config


def test_a_default_config_segments_with_no_extra_keys(tmp_path):
    config_file = write_config(tmp_path, "ws")

    run_segmentation(config_file, "ws")

    container = tmp_path / "vol.zarr"
    assert (container / "rag_ws.db").is_file()

    # one segmentation per default threshold
    segs = sorted(glob.glob(str(container / "segmentations_ws" / "*")))
    assert len(segs) == 3

    params = zarr.open(segs[0], mode="r").attrs["bs_params"]
    assert params["blockwise"] is False
    assert params["block_shape"] == [20, 64, 64]  # one block over the whole array
    assert params["filter_fragments"] == 0.1
    assert params["remove_debris"] == 64


def test_a_whole_array_run_is_the_one_block_run(tmp_path):
    """`block_shape = "roi"` and no blockwise give the same one block."""
    roi_config = write_config(
        tmp_path, "ws", {"blockwise": True, "block_shape": "roi", "num_workers": 4}
    )

    run_segmentation(roi_config, "ws")

    segs = sorted(glob.glob(str(tmp_path / "vol.zarr" / "segmentations_ws" / "*")))
    params = zarr.open(segs[0], mode="r").attrs["bs_params"]
    assert params["blockwise"] is False
    assert params["block_shape"] == [20, 64, 64]
    assert params["context"] == [0, 0, 0]
