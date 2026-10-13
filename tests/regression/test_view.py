"""The viewer binds where it is told, 0.0.0.0 by default (decided on 2026-09-14), and shows
the arrays of a snapshot."""

import contextlib
import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from click.testing import CliRunner

# bootstrapper/__init__.py rebinds the name "view" to the command
view_module = importlib.import_module("bootstrapper.view")


class ViewerStarted(Exception):
    """Raised in place of building a real neuroglancer viewer."""


@pytest.mark.parametrize(
    "args,bind", [([], "0.0.0.0"), (["--bind", "127.0.0.1"], "127.0.0.1")]
)
def test_view_snapshot_binds_the_address(tmp_path, monkeypatch, args, bind):
    def viewer(*args, **kwargs):
        raise ViewerStarted()

    seen = []
    monkeypatch.setattr(
        view_module.neuroglancer, "set_server_bind_address", seen.append
    )
    monkeypatch.setattr(view_module.neuroglancer, "Viewer", viewer)

    result = CliRunner().invoke(view_module.view, ["--snapshot", str(tmp_path), *args])

    assert isinstance(result.exception, ViewerStarted)
    assert seen == [bind]


def test_view_datasets_binds_the_neuroglancer_process(monkeypatch):
    seen = []
    monkeypatch.setattr(
        view_module.subprocess, "run", lambda args, **kw: seen.append(args)
    )

    result = CliRunner().invoke(view_module.view, ["some.zarr/raw"])

    assert result.exit_code == 0, result.exception
    assert seen == [["neuroglancer", "-b", "0.0.0.0", "-d", "some.zarr/raw"]]


def test_view_skips_a_group_in_the_snapshot(tmp_path, monkeypatch):
    path = str(tmp_path / "batch_1.zarr")
    root = zarr.open_group(path, mode="w")
    root.create_group("a_group")
    for name in ["raw", "pred_affs"]:
        root.create_dataset(name, data=np.zeros((1, 4, 8, 8), np.uint8))
        root[name].attrs.update(offset=[0, 0, 0], voxel_size=[40, 8, 8])
    # os.listdir order is arbitrary: put the group first
    listdir = os.listdir
    monkeypatch.setattr(
        os, "listdir", lambda p=".": sorted(listdir(p), key=lambda n: n != "a_group")
    )
    state = SimpleNamespace(layers={}, layout=None)
    monkeypatch.setattr(
        view_module.neuroglancer, "set_server_bind_address", lambda bind: None
    )
    monkeypatch.setattr(
        view_module.neuroglancer,
        "Viewer",
        lambda: SimpleNamespace(txn=lambda: contextlib.nullcontext(state)),
    )

    view_module.view_snapshot(path, "127.0.0.1")

    assert sorted(state.layers) == ["pred_affs", "raw"]
