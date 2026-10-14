"""The 2D LSDs a trainer learns span the same number of voxels at any voxel size, the 10 voxels
that the pretrained 3d_affs_from_2d_* models read: 80 nm at 8 nm voxels, 1680 nm at 168 nm."""

import importlib.util
import json
import sys

import gunpowder as gp
import pytest

from bootstrapper.configs import MODEL_DIR
from bootstrapper.gp import Add2DLSDs


@pytest.mark.parametrize("model", ["2d_lsd", "2d_mtlsd"])
def test_the_lsd_sigma_is_10_voxels_at_any_voxel_size(model, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(f"{MODEL_DIR}/{model}")
    for name in ("model", "unet"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(f"train_{model}", f"{MODEL_DIR}/{model}/train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    with open(f"{MODEL_DIR}/{model}/net_config.json") as f:
        net_config = json.load(f)
    voxels = {}
    for voxel_size in [(40, 8, 8), (400, 168, 168)]:
        pipeline, _, _ = trainer.create_pipeline(voxel_size, net_config, [])
        nodes = []
        pipeline.traverse(lambda p: nodes.append(p.output))
        [lsds] = [n for n in nodes if isinstance(n, Add2DLSDs)]
        voxels[voxel_size] = [s / v for s, v in zip(lsds.sigma[1:], voxel_size[1:])]

    assert voxels == {(40, 8, 8): [10, 10], (400, 168, 168): [10, 10]}
