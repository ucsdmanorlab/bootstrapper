"""A trainer draws its windows only on the labelled sections that bs train finds: each run of
adjacent labelled sections by its share of the labelled voxels, each volume by the sampling key.
Before, RandomLocation drew over the whole box and Reject threw away the draws off the labels:
fluo (1 of 30 frames) took 30 to 42 draws per sample, prism (every 21st of 200 sections) 19."""

import importlib.util
import json
import logging
import os
import sys

import gunpowder as gp
import numpy as np
import pytest
import toml

from bootstrapper.configs import MODEL_DIR
from bootstrapper.train import setup_train
from test_train import UNLABELLED, VOXEL_SIZE, write


@pytest.fixture
def trainer(monkeypatch):
    """Import a model's train.py the way its setup dir runs it."""
    monkeypatch.setattr(logging.getLogger(), "level", logging.getLogger().level)

    def load(model):
        monkeypatch.syspath_prepend(os.path.join(MODEL_DIR, model))
        for name in ("model", "unet"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        spec = importlib.util.spec_from_file_location(f"train_{model}", os.path.join(MODEL_DIR, model, "train.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


def volume(path, sections, shape, patch=None, mask=True):
    """Labels on the given sections; patch (y0, y1, x0, x1) limits the labels of the last section."""
    labels = np.zeros(shape, np.uint32)
    labels[sections] = 1 + np.arange(shape[1] * shape[2], dtype=np.uint32).reshape(shape[1:]) // 600
    if patch:
        y0, y1, x0, x1 = patch
        section = labels[sections[-1]].copy()
        labels[sections[-1]] = 0
        labels[sections[-1], y0:y1, x0:x1] = section[y0:y1, x0:x1]
    sample = {"volume": path.name,
              "raw": write(path / "raw", np.full(shape, 128, np.uint8)),
              "labels": write(path / "labels", labels)}
    if mask:
        sample["mask"] = write(path / "mask", (labels > 0).astype(np.uint8))
    return sample


def create_pipeline(trainer, model, tmp_path, samples, sampling="labels"):
    """bs train on the samples, then the model's pipeline on the worker config it wrote."""
    setup_dir = tmp_path / model
    setup_dir.mkdir()
    (setup_dir / "train.py").touch()
    config = {"setup_dir": str(setup_dir), "voxel_size": list(VOXEL_SIZE), "max_iterations": 10,
              "save_checkpoints_every": 5, "save_snapshots_every": 5, "samples": samples, "sampling": sampling}
    (tmp_path / "01_train.toml").write_text(toml.dumps({"train": {model: config}}))
    worker = toml.load(setup_train(str(tmp_path / "01_train.toml"))[1])

    with open(os.path.join(MODEL_DIR, model, "net_config.json")) as f:
        net_config = json.load(f)
    return trainer(model).create_pipeline(VOXEL_SIZE, net_config, worker["samples"])


def draw(trainer, model, tmp_path, monkeypatch, samples, n, sampling="labels"):
    """n windows from the model's sampling stage: the draws it took, and the first section of
    each accepted window."""
    pipeline, request, _ = create_pipeline(trainer, model, tmp_path, samples, sampling)
    stages = []
    pipeline.traverse(lambda p: stages.append(p))
    [sampling_stage] = [p for p in stages if isinstance(p.output, gp.RandomProvider)]

    draws = []
    process = gp.RandomLocation.process

    def record(self, batch, request):
        draws.append(batch[UNLABELLED].spec.roi.begin[0] // VOXEL_SIZE[0])
        return process(self, batch, request)

    monkeypatch.setattr(gp.RandomLocation, "process", record)
    accepted = []
    with gp.build(sampling_stage):
        for _ in range(n):
            window = gp.BatchRequest()  # no seed: Reject would ask again for the same window
            for key, spec in request.items():
                if key in sampling_stage.spec:
                    window[key] = spec.copy()
            sampling_stage.request_batch(window)
            accepted.append(draws[-1])
    return len(draws), accepted


@pytest.mark.parametrize("model", ["2d_affs", "2d_lsd", "2d_mtlsd"])
@pytest.mark.parametrize("depth,sections", [(30, [15]), (200, list(range(10, 200, 21)))], ids=["fluo", "prism"])
def test_labels_on_few_sections_of_a_deep_box_cost_under_2_draws_per_sample(
    trainer, model, depth, sections, tmp_path, monkeypatch
):
    sample = volume(tmp_path / "v.zarr", sections, (depth, 112, 112))

    draws, accepted = draw(trainer, model, tmp_path, monkeypatch, [sample], 6)

    assert draws / 6 < 2, f"{draws} draws for 6 samples"
    assert set(accepted) <= set(sections)


@pytest.mark.parametrize("model", ["3d_affs", "3d_lsd", "3d_mtlsd"])
def test_a_3d_window_touches_a_labelled_run_at_either_end_of_the_box(trainer, model, tmp_path, monkeypatch):
    """No mask: the labels mark the labelled voxels. The output window is 4 sections deep."""
    sample = volume(tmp_path / "v.zarr", [0, 1, 198, 199], (200, 112, 112), mask=False)

    draws, accepted = draw(trainer, model, tmp_path, monkeypatch, [sample], 4)

    assert draws / 4 < 2, f"{draws} draws for 4 samples"
    assert set(accepted) <= {0, 1, 195, 196}


def test_sections_are_drawn_by_labelled_voxels_and_volumes_by_the_sampling_key(trainer, tmp_path, monkeypatch):
    """In a, sections 10 to 12 hold 3 x 208 x 208 labelled voxels and section 40 a 52 x 52 patch: 98 % and 2 %
    of a. b labels section 25. sampling = "equal" gives each volume half. Before, a window that held 5 % labels
    was as good as any, and section 40 took a quarter of a."""
    a = volume(tmp_path / "a.zarr", [10, 11, 12, 40], (50, 208, 208), patch=(78, 130, 78, 130))
    b = volume(tmp_path / "b.zarr", [25], (50, 208, 208))
    n = 300

    _, accepted = draw(trainer, "2d_affs", tmp_path, monkeypatch, [a, b], n, sampling="equal")

    share = {z: accepted.count(z) / n for z in (10, 11, 12, 40, 25)}
    assert share[40] == pytest.approx(0.5 * 0.0625 / 3.0625, abs=0.03), share
    assert share[10] + share[11] + share[12] == pytest.approx(0.5 * 3 / 3.0625, abs=0.1), share
    assert share[25] == pytest.approx(0.5, abs=0.1), share


@pytest.mark.parametrize("model", ["2d_affs", "2d_lsd", "2d_mtlsd", "3d_affs", "3d_lsd", "3d_mtlsd"])
@pytest.mark.parametrize("sections,mask", [([10], True), ([9, 10, 11], False)], ids=["one_section", "short_run"])
def test_the_full_pipeline_builds_a_batch_from_a_short_labelled_run(trainer, model, sections, mask, tmp_path):
    """The LSD and affinity nodes read the mask with z context past the output window (the 3D LSD
    reads 12 sections for a 4-section output), so the crop to a labelled run holds only the sampling mask."""
    sample = volume(tmp_path / "v.zarr", sections, (20, 384, 384), mask=mask)

    pipeline, request, keys = create_pipeline(trainer, model, tmp_path, [sample])
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    for name, key in keys.items():
        data = batch[key].data
        if name.endswith("_weights"):
            assert np.count_nonzero(data) > 0, f"{name} is all zero: the window missed the labels"
