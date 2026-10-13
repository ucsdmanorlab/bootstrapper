"""Training must use only the granted GPUs and say so when it runs on the CPU; -s is the step,
and a model script names both dirs when the setup_dir is not its own. bs train refuses a sample
whose labels reach past the raw or whose mask reaches past the labels, and names the bs prepare
command when the setup or a mask it makes is missing. It draws the volumes by labelled voxels,
equally, inversely or by your numbers. The shipped trainers draw by them, deform within
sections, keep raw on labels and train on labels."""

import ast
import importlib
import importlib.util
import inspect
import json
import logging
import os
import runpy
import sys

import click
import gunpowder as gp
import numpy as np
import pytest
import toml
import torch
from click.testing import CliRunner
from funlib.persistence import Array, prepare_ds

import bootstrapper.training as training
from bootstrapper.configs import MODEL_DIR, MODEL_NAMES
from bootstrapper.train import setup_train

# bootstrapper/__init__.py rebinds the name "train" to the command
train_module = importlib.import_module("bootstrapper.train")

VOXEL_SIZE = gp.Coordinate(50, 4, 4)


def write(path, data, offset=(0, 0, 0)):
    """An array of 50 x 4 x 4 nm voxels; offset in voxels."""
    ds = prepare_ds(
        str(path),
        data.shape,
        offset=VOXEL_SIZE * gp.Coordinate(offset),
        voxel_size=VOXEL_SIZE,
        dtype=data.dtype,
    )
    ds[ds.roi] = data
    return str(path)


def write_config(tmp_path, arrays=None, **extra):
    """A one-step train file, [train.setup_01], with one sample in data.zarr; arrays maps a
    name to its (offset, shape) in voxels."""
    arrays = arrays or {"raw": ((0, 0, 0), (4, 4, 4)), "labels": ((0, 0, 0), (4, 4, 4))}
    setup_dir = tmp_path / "setup_01"
    setup_dir.mkdir()
    (setup_dir / "train.py").touch()
    sample = {
        name: write(tmp_path / "data.zarr" / name, np.zeros(shape, np.uint8), offset)
        for name, (offset, shape) in arrays.items()
    }

    config = {
        "setup_dir": str(setup_dir),
        "voxel_size": list(VOXEL_SIZE),
        "max_iterations": 10,
        "save_checkpoints_every": 5,
        "save_snapshots_every": 5,
        "samples": [sample],
    }
    config.update(extra)
    config_file = tmp_path / "01_train.toml"
    config_file.write_text(toml.dumps({"train": {"setup_01": config}}))
    return str(config_file)


class FakeTrainer:
    """Records what fit() asks the Trainer for, without training."""

    kwargs = {}

    def __init__(self, **kwargs):
        FakeTrainer.kwargs = kwargs

    def fit(self, *args, **kwargs):
        pass


class EmptyDataset(training.IterableDataset):
    def __iter__(self):
        return iter([])


@pytest.mark.parametrize("devices,expected", [(None, 1), ([0], [0])])
def test_fit_passes_devices_to_trainer(tmp_path, monkeypatch, devices, expected):
    monkeypatch.setattr(training.pl, "Trainer", FakeTrainer)

    training.fit(
        object(),
        EmptyDataset(),
        str(tmp_path),
        max_iterations=10,
        save_checkpoints_every=5,
        snapshot_callback=object(),
        num_workers=0,
        devices=devices,
    )

    assert FakeTrainer.kwargs["devices"] == expected


class RecordingLoader:
    kwargs = {}

    def __init__(self, dataset, **kwargs):
        RecordingLoader.kwargs = kwargs


@pytest.mark.parametrize("affinity", [True, False])
@pytest.mark.parametrize("cpus,workers", [(96, 32), (8, 4), (1, 1)])
def test_fit_takes_half_the_cpus_as_data_workers_up_to_32(tmp_path, monkeypatch, affinity, cpus, workers):
    monkeypatch.setattr(training.pl, "Trainer", FakeTrainer)
    monkeypatch.setattr(training, "DataLoader", RecordingLoader)
    monkeypatch.setattr(training.os, "cpu_count", lambda: cpus if not affinity else 1000)
    if affinity:
        monkeypatch.setattr(training.os, "sched_getaffinity", lambda pid: set(range(cpus)), raising=False)
    else:
        monkeypatch.delattr(training.os, "sched_getaffinity", raising=False)

    training.fit(object(), EmptyDataset(), str(tmp_path), 10, 5, object())

    assert RecordingLoader.kwargs["num_workers"] == workers


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_script_accepts_devices(model_name):
    script = os.path.join(MODEL_DIR, model_name, "train.py")
    tree = ast.parse(open(script).read(), script)
    train_fn = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"
    ][0]

    assert "devices" in [a.arg for a in train_fn.args.args]


def test_devices_reach_the_worker_config(tmp_path):
    config_file = write_config(tmp_path, devices=[0])

    train_script, worker_config = setup_train(config_file)

    assert train_script == str(tmp_path / "setup_01" / "train.py")
    assert worker_config == str(tmp_path / "setup_01" / "train.toml")
    assert toml.load(worker_config)["devices"] == [0]


def test_overrides_reach_the_worker_config_and_leave_the_file(tmp_path):
    config_file = write_config(tmp_path)

    _, worker_config = setup_train(config_file, max_iterations=99)

    assert toml.load(worker_config)["max_iterations"] == 99
    assert toml.load(config_file)["train"]["setup_01"]["max_iterations"] == 10


def test_missing_labels_is_refused_by_path(tmp_path):
    config_file = write_config(tmp_path)
    config = toml.load(config_file)
    absent = str(tmp_path / "data.zarr" / "absent")
    config["train"]["setup_01"]["samples"][0]["labels"] = absent
    open(config_file, "w").write(toml.dumps(config))

    with pytest.raises(ValueError, match="labels") as error:
        setup_train(config_file)

    assert absent in str(error.value)


@pytest.mark.parametrize(
    "setup,mask,message",
    [
        (
            "2d_mtlsd",
            "data.zarr/labels",
            "{0}/2d_mtlsd has no train.py; bs prepare {0} copies the model scripts there",
        ),
        (
            "setup_01",
            "training_mask.zarr/v",
            "training mask {0}/training_mask.zarr/v does not exist; bs prepare {0} makes it",
        ),
        (
            "setup_01",
            "data.zarr/absent",
            "training mask {0}/data.zarr/absent does not exist",
        ),
    ],
)
def test_a_missing_setup_or_mask_is_refused_with_its_fix(
    tmp_path, setup, mask, message
):
    config_file = write_config(tmp_path, setup_dir=str(tmp_path / setup))
    config = toml.load(config_file)
    config["train"]["setup_01"]["samples"][0]["mask"] = str(tmp_path / mask)
    open(config_file, "w").write(toml.dumps(config))

    with pytest.raises(click.ClickException) as error:
        setup_train(config_file)

    assert error.value.message == message.format(tmp_path)


BIG = ((0, 0, 0), (8, 192, 192))
CROP = ((0, 32, 32), (8, 128, 128))
CORNER = ((0, 0, 0), (8, 16, 16))


@pytest.mark.parametrize(
    "arrays,refused,message",
    [
        (
            {"raw": CROP, "labels": BIG},
            "labels",
            "covers [0:400, 0:768, 0:768] (400, 768, 768) nm, past the raw at [0:400, 128:640, 128:640] "
            "(400, 512, 512) nm; crop the labels to [0:400, 128:640, 128:640] (400, 512, 512) nm",
        ),
        (
            {"raw": BIG, "labels": CROP, "mask": BIG},
            "mask",
            "covers [0:400, 0:768, 0:768] (400, 768, 768) nm, past the labels at [0:400, 128:640, 128:640] "
            "(400, 512, 512) nm; crop the mask to [0:400, 128:640, 128:640] (400, 512, 512) nm",
        ),
        (
            {"raw": BIG, "labels": CROP, "mask": CORNER},
            "mask",
            "covers [0:400, 0:64, 0:64] (400, 64, 64) nm, past the labels at [0:400, 128:640, 128:640] "
            "(400, 512, 512) nm; they do not overlap",
        ),
    ],
)
def test_an_array_past_its_container_is_refused_with_the_crop(
    tmp_path, arrays, refused, message
):
    config_file = write_config(tmp_path, arrays)

    with pytest.raises(click.ClickException) as error:
        setup_train(config_file)

    path = tmp_path / "data.zarr" / refused
    assert error.value.message == f"{refused} {path} {message}"


def test_a_mask_inside_the_labels_is_accepted(tmp_path):
    mask = ((0, 48, 48), (8, 64, 64))
    config_file = write_config(tmp_path, {"raw": BIG, "labels": CROP, "mask": mask})

    assert setup_train(config_file)[1] == str(tmp_path / "setup_01" / "train.toml")


def two_volumes(tmp_path):
    """a has 48 labelled voxels; b has 64, but its training mask holds 16."""
    data, section = np.ones((4, 4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8)
    section[0] = 1
    a, b = ({"volume": n, "raw": write(tmp_path / f"{n}.zarr" / "raw", data)} for n in "ab")
    a["labels"] = write(tmp_path / "a.zarr" / "labels", data - section)
    b["labels"] = write(tmp_path / "b.zarr" / "labels", data)
    b["mask"] = write(tmp_path / "b.zarr" / "mask", section)
    return [a, b]


@pytest.mark.parametrize(
    "sampling,expected",
    [(None, [0.75, 0.25]), ("equal", [0.5, 0.5]), ("inv", [0.25, 0.75]), ([1, 4], [0.2, 0.8]), ({"b": 4, "a": 1}, [0.2, 0.8])],
)
def test_volumes_are_drawn_by_labelled_voxels_equally_inversely_or_as_given(tmp_path, capsys, sampling, expected):
    extra = {} if sampling is None else {"sampling": sampling}
    config_file = write_config(tmp_path, samples=two_volumes(tmp_path), **extra)

    worker_config = toml.load(setup_train(config_file)[1])

    assert [s["probability"] for s in worker_config["samples"]] == pytest.approx(expected)
    assert "sampling" not in worker_config
    assert capsys.readouterr().out == f"Sampling a {expected[0]:.0%}, b {expected[1]:.0%}\n"


@pytest.mark.parametrize(
    "sampling,message",
    [
        ("uniform", 'sampling = "uniform" names no scheme; give "labels", "equal", "inv", or numbers such as [3, 1]'),
        ([1], "sampling needs one number per volume, as a list in the order a, b or a table by name"),
        ({"a": 1}, "sampling needs one number per volume, as a list in the order a, b or a table by name"),
    ],
)
def test_a_sampling_that_fits_no_volume_is_refused_with_the_fix(tmp_path, sampling, message):
    config_file = write_config(tmp_path, samples=two_volumes(tmp_path), sampling=sampling)

    with pytest.raises(click.ClickException) as error:
        setup_train(config_file)

    assert error.value.message == message


@pytest.mark.parametrize(
    "grant,why",
    [
        ("", "CUDA_VISIBLE_DEVICES is set but empty"),
        (None, "no CUDA device is visible"),
    ],
)
def test_cpu_training_warns_with_the_reason(monkeypatch, grant, why):
    monkeypatch.setattr(training.torch.cuda, "is_available", lambda: False)
    if grant is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", grant)

    assert why in training.device_warning()


def test_train_s_is_the_step(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        train_module, "run_training", lambda config, step, **kw: seen.update(step=step)
    )
    config = tmp_path / "train.toml"
    config.write_text("")

    result = CliRunner().invoke(train_module.train, [str(config), "-s", "apical"])

    assert result.exit_code == 0, result.output
    assert seen["step"] == "apical"


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_a_model_script_names_both_dirs_when_setup_dir_differs(
    model_name, tmp_path, monkeypatch
):
    script = os.path.join(MODEL_DIR, model_name, "train.py")
    setup_dir = os.path.realpath(os.path.dirname(script))
    config = tmp_path / "train.toml"
    config.write_text('setup_dir = "/elsewhere"\n')
    monkeypatch.setattr(sys, "argv", ["train.py", str(config)])
    monkeypatch.syspath_prepend(setup_dir)
    monkeypatch.delitem(sys.modules, "model", raising=False)
    monkeypatch.setattr(logging.getLogger(), "level", logging.getLogger().level)

    with pytest.raises(SystemExit) as stop:
        runpy.run_path(script, run_name="__main__")

    assert "/elsewhere" in str(stop.value) and setup_dir in str(stop.value)


REAL_DATA = ["2d_affs", "2d_lsd", "2d_mtlsd", "3d_affs", "3d_lsd", "3d_mtlsd"]
RAW, LABELS = gp.ArrayKey("RAW"), gp.ArrayKey("LABELS")
UNLABELLED = gp.ArrayKey("UNLABELLED")


@pytest.fixture
def trainer(monkeypatch):
    """Import a model's train.py the way its setup dir runs it."""
    monkeypatch.setattr(logging.getLogger(), "level", logging.getLogger().level)

    def load(model):
        monkeypatch.syspath_prepend(os.path.join(MODEL_DIR, model))
        for name in ("model", "unet"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        spec = importlib.util.spec_from_file_location(
            f"train_{model}", os.path.join(MODEL_DIR, model, "train.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


def build(module, model, samples, **shapes):
    with open(os.path.join(MODEL_DIR, model, "net_config.json")) as f:
        net_config = json.load(f)
    return module.create_pipeline(VOXEL_SIZE, {**net_config, **shapes}, samples)


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_trainer_imports_and_its_loss_runs(trainer, model):
    loss = trainer(model).WeightedMSELoss()
    pred, gt, weights = torch.zeros(1, 2, 2), torch.ones(1, 2, 2), torch.ones(1, 2, 2)
    terms = len(inspect.signature(loss.forward).parameters) // 3

    assert float(loss(*[pred, gt, weights] * terms)) == terms


@pytest.fixture(scope="module")
def ramp():
    """Labels hold their own source voxel (1 + (z * Y + y) * X + x), raw holds (x + 1) / (X + 1)."""
    Z, Y, X = 36, 384, 384
    z, y, x = np.ogrid[:Z, :Y, :X]
    labels = (1 + (z * Y + y) * X + x).astype(np.uint32)
    raw = np.broadcast_to((x + 1) / (X + 1), labels.shape).astype(np.float32)
    return Array(raw, voxel_size=VOXEL_SIZE), Array(labels, voxel_size=VOXEL_SIZE)


@pytest.mark.parametrize("model", REAL_DATA)
def test_deform_stays_in_sections_and_keeps_raw_on_labels(
    trainer, model, ramp, tmp_path
):
    tiny = write(tmp_path / "tiny.zarr" / "a", np.ones((1, 1, 1), np.uint8))
    pipeline, request, _ = build(trainer(model), model, [{"raw": tiny, "labels": tiny, "probability": 1}])
    nodes = []
    pipeline.traverse(lambda p: nodes.append(p.output))
    [deform] = [n for n in nodes if isinstance(n, gp.DeformAugment)]
    deform.p = 1.0
    raw, labels = ramp
    pipeline = (
        (gp.ArraySource(RAW, raw, True), gp.ArraySource(LABELS, labels, False))
        + gp.MergeProvider()
        + gp.Pad(RAW, None)
        + gp.Pad(LABELS, None)
        + deform
    )
    shift = (raw.roi.center - request[RAW].roi.center) // VOXEL_SIZE * VOXEL_SIZE
    Z, Y, X = labels.data.shape

    with gp.build(pipeline):
        for seed in range(2):
            draw = gp.BatchRequest(random_seed=seed)
            for key in (RAW, LABELS):
                draw[key] = gp.ArraySpec(roi=request[key].roi.shift(shift))
            batch = pipeline.request_batch(draw)

            source = batch[LABELS].data.astype(np.int64) - 1
            z, x = source // (Y * X), source % X
            first = batch[LABELS].spec.roi.begin[0] // VOXEL_SIZE[0]
            sections = np.arange(first, first + len(z))[:, None, None]
            on_labels = batch[RAW].crop(batch[LABELS].spec.roi).data * (X + 1) - 1

            offset = float(np.abs(on_labels - x).max())

            assert (z == sections).all(), "a section samples other sections"
            assert min(len(np.unique(s)) for s in x) > 1, (
                "a section collapses to one source point"
            )
            assert offset <= 0.5 + 1e-3, f"raw is {offset:.1f} voxels off the labels"


@pytest.mark.parametrize("model", REAL_DATA)
def test_a_trainer_draws_each_sample_by_its_probability(trainer, model, tmp_path):
    tiny = write(tmp_path / "tiny.zarr" / "a", np.ones((1, 1, 1), np.uint8))
    samples = [{"raw": tiny, "labels": tiny, "probability": p} for p in (0.75, 0.25)]
    pipeline, _, _ = build(trainer(model), model, samples)
    nodes = []
    pipeline.traverse(lambda p: nodes.append(p.output))
    [draw] = [n for n in nodes if isinstance(n, gp.RandomProvider)]

    assert draw.probabilities == [0.75, 0.25]


@pytest.fixture(scope="module")
def sparse_sample(tmp_path_factory):
    """One labelled section in 12: a source window passes the 5 % check, an output window often has none."""
    path = tmp_path_factory.mktemp("sparse") / "sample.zarr"
    Z, Y, X = 48, 256, 256
    y, x = np.ogrid[:Y, :X]
    labels = np.broadcast_to(1 + (y // 24) * 100 + x // 24, (Z, Y, X)).astype(np.uint32)
    mask = np.zeros((Z, Y, X), np.uint8)
    mask[::12] = 1
    return {
        "raw": write(path / "raw", np.full((Z, Y, X), 128, np.uint8)),
        "labels": write(path / "labels", labels),
        "mask": write(path / "mask", mask),
        "probability": 1,
    }


@pytest.mark.parametrize("model", ["3d_affs", "3d_lsd", "3d_mtlsd"])
def test_every_output_window_is_labelled(trainer, model, sparse_sample):
    small = {"input_shape": [12, 60, 60], "output_shape": [4, 20, 20]}
    pipeline, request, _ = build(trainer(model), model, [sparse_sample], **small)
    request[UNLABELLED] = request[LABELS].copy()

    with gp.build(pipeline):
        for _ in range(5):
            labelled = pipeline.request_batch(request)[UNLABELLED].data.mean()
            assert labelled > 0.05, f"an output window is {labelled:.0%} labelled"


@pytest.mark.parametrize("model_name", REAL_DATA)
def test_model_script_leaves_the_data_workers_to_fit(model_name):
    script = os.path.join(MODEL_DIR, model_name, "train.py")
    calls = [n for n in ast.walk(ast.parse(open(script).read(), script))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "fit"]

    assert calls and all("num_workers" not in [k.arg for k in c.keywords] for c in calls)
