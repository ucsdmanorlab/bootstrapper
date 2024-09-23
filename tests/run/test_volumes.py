import pytest
from unittest.mock import patch, MagicMock
import os
from bootstrapper.run.volumes import (
    process_zarr,
    process_non_zarr,
    process_dataset,
    prepare_volume,
    make_volumes,
)


@pytest.fixture
def mock_zarr():
    with patch("bootstrapper.run.volumes.zarr") as mock:
        yield mock


@pytest.fixture
def mock_click():
    with patch("bootstrapper.run.volumes.click") as mock:
        yield mock


def test_process_zarr(mock_zarr, mock_click):
    mock_zarr.open.return_value = MagicMock()
    mock_click.prompt.side_effect = ["input_ds", "output_ds"]
    mock_click.confirm.return_value = False

    result = process_zarr("input.zarr", "output.zarr", "raw")

    assert result == (
        "output.zarr/output_ds",
        mock_zarr.open.return_value["input_ds"].attrs["voxel_size"],
    )
    mock_zarr.open.assert_any_call("input.zarr")
    mock_zarr.open.assert_any_call("output.zarr", "a")
    mock_click.prompt.assert_called()


def test_process_non_zarr(mock_click):
    mock_click.prompt.side_effect = [
        "output_ds",
        "uint8",
        "1 1 1",
        "0 0 0",
        "z y x",
        "nm nm nm",
    ]
    mock_click.confirm.return_value = False

    with patch("bootstrapper.run.volumes.subprocess.run") as mock_run:
        result = process_non_zarr("input.tif", "output.zarr", "raw")

    assert result == ("output.zarr/output_ds", (1, 1, 1))
    mock_run.assert_called()


def test_process_dataset(mock_click):
    mock_click.confirm.side_effect = [False, False, False]

    with patch("bootstrapper.run.volumes.process_zarr") as mock_process_zarr:
        mock_process_zarr.return_value = ("output.zarr/dataset", (1, 1, 1))
        result = process_dataset("input.zarr", "output.zarr", "raw")

    assert result == ("output.zarr/dataset", None, (1, 1, 1))


def test_prepare_volume(mock_click):
    mock_click.prompt.side_effect = [
        "output.zarr",
        "raw_input.zarr",
        "labels_input.zarr",
    ]

    with patch("bootstrapper.run.volumes.process_dataset") as mock_process_dataset:
        mock_process_dataset.side_effect = [
            ("output.zarr/raw", None, (1, 1, 1)),
            ("output.zarr/labels", None, (1, 1, 1)),
        ]
        with patch("os.path.abspath") as mock_abspath:
            mock_abspath.side_effect = lambda x: (
                f"/base_dir/{x}" if not x.startswith("/base_dir") else x
            )
            result = prepare_volume("/base_dir", 0)

    expected = {
        "zarr_container": "/base_dir/output.zarr",
        "raw_dataset": "/base_dir/output.zarr/raw",
        "raw_mask_dataset": None,
        "labels_dataset": "/base_dir/output.zarr/labels",
        "labels_mask_dataset": None,
        "voxel_size": [1, 1, 1],
    }
    assert result == expected


def test_make_volumes(mock_click):
    mock_click.prompt.return_value = 2

    with patch("bootstrapper.run.volumes.prepare_volume") as mock_prepare_volume:
        mock_prepare_volume.side_effect = [
            {"zarr_container": "volume_1.zarr"},
            {"zarr_container": "volume_2.zarr"},
        ]
        result = make_volumes("/base_dir")

    assert len(result) == 2
    assert result[0]["zarr_container"] == "volume_1.zarr"
    assert result[1]["zarr_container"] == "volume_2.zarr"
