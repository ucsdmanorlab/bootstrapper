"""A raw with channels (c z y x), such as 18-channel PRISM light microscopy, trains: each
trainer that reads raw gives the model a batch of in_channels channels."""

import importlib.util
import json
import shutil
import sys

import gunpowder as gp
import numpy as np
import pytest
import torch
from funlib.persistence import prepare_ds

from bootstrapper.configs import MODEL_DIR

RAW_MODELS = ["2d_affs", "2d_lsd", "2d_mtlsd", "3d_affs", "3d_lsd", "3d_mtlsd"]


def write(path, data, axis_names=("z", "y", "x")):
    ds = prepare_ds(str(path), shape=data.shape, offset=(0, 0, 0), voxel_size=(50, 8, 8),
                    axis_names=list(axis_names), units=["nm"] * 3, dtype=data.dtype)
    ds[ds.roi] = data
    return str(path)


@pytest.mark.parametrize("model", RAW_MODELS)
def test_a_trainer_gives_the_model_a_raw_with_channels(model, tmp_path, monkeypatch):
    setup = tmp_path / model
    shutil.copytree(f"{MODEL_DIR}/{model}", setup)
    net_config = json.loads((setup / "net_config.json").read_text())
    net_config["in_channels"] = 2
    (setup / "net_config.json").write_text(json.dumps(net_config))
    monkeypatch.syspath_prepend(str(setup))
    for name in ("model", "unet"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(f"train_{model}", setup / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    shape = (40, 400, 400)
    labels = np.ones(shape, np.uint64)
    labels[:, :, 200:] = 2
    raw = write(tmp_path / "data.zarr" / "raw", np.random.default_rng(0).integers(0, 255, (2, *shape), np.uint8),
                ("c^", "z", "y", "x"))
    sample = {"raw": raw, "labels": write(tmp_path / "data.zarr" / "labels", labels), "probability": 1.0}

    pipeline, request, keys = trainer.create_pipeline((50, 8, 8), net_config, [sample])
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)
    with torch.no_grad():
        outputs = trainer.LitModel().model(torch.from_numpy(batch[keys["raw"]].data))

    assert batch[keys["raw"]].data.shape[1] == 2
    assert all(torch.isfinite(o).all() for o in outputs)


def test_defect_augment_lowers_the_contrast_of_a_section_in_every_channel():
    """Section z of channel c holds 0.1 z + 0.3 c plus a ramp in x: the section's mean over all
    channels, not one channel's mean over all sections, stays."""
    from funlib.persistence import Array

    from bootstrapper.gp import DefectAugment

    raw_key = gp.ArrayKey("RAW")
    c, z, x = np.ogrid[:2, :4, :8]
    data = np.ascontiguousarray(np.broadcast_to((0.1 * z + 0.3 * c + 0.01 * x)[:, :, None, :], (2, 4, 8, 8)), np.float32)
    source = gp.ArraySource(raw_key, Array(data, voxel_size=(50, 8, 8)), True)
    pipeline = source + DefectAugment(raw_key, prob_missing=0.0, prob_low_contrast=1.0, contrast_scale=0.5)
    request = gp.BatchRequest()
    request[raw_key] = gp.ArraySpec(roi=gp.Roi((0, 0, 0), (200, 64, 64)))

    with gp.build(pipeline):
        out = pipeline.request_batch(request)[raw_key].data

    mean = data.mean(axis=(0, 2, 3), keepdims=True)
    np.testing.assert_allclose(out, (data - mean) * 0.5 + mean, atol=1e-6)


@pytest.mark.parametrize("model", ["3d_affs", "3d_lsd", "3d_mtlsd"])
def test_a_3d_predict_reads_a_raw_with_channels(model, tmp_path, monkeypatch):
    setup = tmp_path / model
    shutil.copytree(f"{MODEL_DIR}/{model}", setup)
    net_config = json.loads((setup / "net_config.json").read_text())
    net_config["in_channels"] = 2
    net_config["shape_increase"] = [0, 0, 0]
    (setup / "net_config.json").write_text(json.dumps(net_config))
    monkeypatch.syspath_prepend(str(setup))
    for name in ("model", "unet"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(f"predict_{model}", setup / "predict.py")
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    torch.save(predictor.Model().state_dict(), tmp_path / "model_checkpoint_0")
    raw = write(tmp_path / "data.zarr" / "raw", np.random.default_rng(0).integers(0, 255, (2, 8, 104, 104), np.uint8),
                ("c^", "z", "y", "x"))
    outputs = []
    for name, output in net_config["outputs"].items():
        ds = prepare_ds(str(tmp_path / "pred.zarr" / name), shape=(output["dims"], 8, 104, 104), offset=(0, 0, 0),
                        voxel_size=(50, 8, 8), axis_names=["c^", "z", "y", "x"], units=["nm"] * 3, dtype=np.uint8)
        outputs.append(ds)

    predictor.predict.callback(checkpoint=str(tmp_path / "model_checkpoint_0"), input_datasets=[raw],
                               output_datasets=[str(tmp_path / "pred.zarr" / n) for n in net_config["outputs"]],
                               roi_offset="0 0 0", roi_shape="200 832 832", num_workers=1, daisy=False)

    assert all(ds[ds.roi][:, :4].any() for ds in outputs)
