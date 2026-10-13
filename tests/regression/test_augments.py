"""Regression tests for the training-time augmentation defects fixed in v1.0."""

import json
import random
from pathlib import Path

import gunpowder as gp
import numpy as np
import pytest
from funlib.persistence import Array

import bootstrapper.gp as bsgp
from bootstrapper.gp import (
    CreateLabels,
    DefectAugment,
    ImpulseNoiseAugment,
    ObfuscateLabels,
)

MODEL_DIR = Path(bsgp.__file__).parent.parent / "models"

RAW = gp.ArrayKey("RAW")
LABELS = gp.ArrayKey("LABELS")
OBFUSCATED_LABELS = gp.ArrayKey("OBFUSCATED_LABELS")


def test_impulse_noise_corrupts_pixel_p_of_the_pixels():
    spec = gp.ArraySpec(
        roi=gp.Roi((0, 0, 0), (10, 200, 200)),
        voxel_size=(1, 1, 1),
        interpolatable=True,
    )
    batch = gp.Batch()
    batch[RAW] = gp.Array(np.full((10, 200, 200), -1.0, dtype=np.float32), spec)

    np.random.seed(0)
    ImpulseNoiseAugment(RAW, pixel_p=0.05, p=0.5).process(batch, gp.BatchRequest())

    corrupted = float(np.mean(batch[RAW].data != -1.0))
    assert corrupted == pytest.approx(0.05, abs=0.005)


@pytest.mark.parametrize(
    "model", sorted(p.parent.name for p in MODEL_DIR.glob("*/net_config.json"))
)
def test_net_config_dims_match_the_neighborhood(model):
    net_config = json.loads((MODEL_DIR / model / "net_config.json").read_text())

    for name, output in net_config["outputs"].items():
        if "neighborhood" in output:
            assert output["dims"] == len(output["neighborhood"]), name


def _generate_labels(**kwargs):
    node = CreateLabels(LABELS, anisotropy_range=(2, 4), shape=(10, 60, 60), **kwargs)
    random.seed(0)
    np.random.seed(0)
    return node._generate_labels((slice(0, 10), slice(0, 60), slice(0, 60)))


def test_create_labels_repeats_for_one_seed():
    assert np.array_equal(_generate_labels(), _generate_labels())


def test_create_labels_honours_dtype():
    assert _generate_labels(dtype=np.uint64).dtype == np.uint64


def test_create_labels_takes_a_background_fraction():
    none = _generate_labels(background_prob=0.0)
    every = _generate_labels(background_prob=1.0)

    assert np.mean(every == 0) > np.mean(none == 0)
    assert np.array_equal(_generate_labels(), _generate_labels(background_prob=0.2))


def test_obfuscate_labels_draws_each_operation_on_its_own():
    node = ObfuscateLabels(
        LABELS, OBFUSCATED_LABELS, num_tries=1, p_split=0.1, p_merge=0.1, p_artifact=0.1
    )

    random.seed(0)
    tries = 20000
    both = sum(
        {"split", "merge"} <= set(node._generate_operations()) for _ in range(tries)
    )

    assert both / tries == pytest.approx(0.1 * 0.1, abs=0.005)


def test_defect_augment_requests_the_roi_the_deform_needs():
    array = Array(
        np.random.random((40, 400, 400)).astype(np.float32),
        offset=(0, 0, 0),
        voxel_size=(1, 1, 1),
    )
    pipeline = (
        gp.ArraySource(RAW, array, True)
        + gp.Pad(RAW, None)
        + DefectAugment(RAW, prob_missing=0.0, prob_low_contrast=0.0, prob_deform=1.0)
    )

    request = gp.BatchRequest()
    request[RAW] = gp.ArraySpec(roi=gp.Roi((10, 100, 100), (10, 100, 100)))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    assert batch[RAW].data.shape == (10, 100, 100)
