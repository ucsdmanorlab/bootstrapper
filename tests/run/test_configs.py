import pytest
from unittest.mock import patch, MagicMock
import yaml
import numpy as np
from bootstrapper.run.configs import (
    check_and_update,
    save_config,
    copy_model_scripts,
    get_rag_db_config,
    choose_model,
    create_training_config,
    create_prediction_configs,
    create_segmentation_configs,
    create_evaluation_configs,
    create_filter_configs,
    make_round_configs,
    get_roi,
    make_configs,
)
from shutil import rmtree


@pytest.fixture
def mock_click():
    with patch("bootstrapper.run.configs.click") as mock:
        yield mock


@pytest.fixture
def mock_open():
    with patch("builtins.open", create=True) as mock:
        yield mock


def test_check_and_update(mock_click):
    mock_click.confirm.return_value = True
    config = {"key": "value"}
    result = check_and_update(config)
    assert result == config

    mock_click.confirm.return_value = False
    mock_click.edit.return_value = yaml.dump({"new_key": "new_value"})
    result = check_and_update(config)
    assert result == {"new_key": "new_value"}


def test_save_config(mock_open):
    config = {"key": "value"}
    with patch("yaml.dump") as mock_yaml_dump:
        save_config(config, "test.yaml")
        mock_yaml_dump.assert_called_once_with(config, mock_open().__enter__())


def test_copy_model_scripts(mock_open):
    with patch("bootstrapper.run.configs.copytree") as mock_copytree:
        copy_model_scripts("test_model", "setup_dir")
        mock_copytree.assert_called_once()


def test_get_rag_db_config_sqlite(mock_click):
    mock_click.prompt.side_effect = ["nodes", "edges", "db.sqlite"]
    result = get_rag_db_config("db.sqlite")
    assert result == {
        "db_file": "db.sqlite",
        "nodes_table": "nodes",
        "edges_table": "edges",
    }


def test_get_rag_db_config_pgsql(mock_click):
    mock_click.prompt.side_effect = [
        "nodes",
        "edges",
        "host",
        "user",
        "password",
        5432,
        "db_name",
    ]
    with patch.dict("os.environ", {}, clear=True):
        result = get_rag_db_config()
    assert result == {
        "db_host": "host",
        "db_user": "user",
        "db_password": "password",
        "db_port": 5432,
        "db_name": "db_name",
        "nodes_table": "nodes",
        "edges_table": "edges",
    }


def test_choose_model(mock_click):
    mock_click.prompt.return_value = "test_model"
    with patch(
        "bootstrapper.run.configs.os.listdir",
        return_value=["test_model", "another_model"],
    ):
        with patch("bootstrapper.run.configs.os.path.isdir", return_value=True):
            result = choose_model(0, "round_1")
    assert result == "test_model"


def test_create_training_config(mock_click):
    mock_click.prompt.side_effect = [10, 30001, 5000, 1000]
    volumes = [
        {
            "zarr_container": "test.zarr",
            "raw_dataset": "raw",
            "labels_dataset": "labels",
            "labels_mask_dataset": "mask",
            "voxel_size": [1, 1, 1],
        }
    ]
    with patch("bootstrapper.run.configs.copy_model_scripts"):
        with patch("bootstrapper.run.configs.check_and_update", return_value={}):
            with patch("bootstrapper.run.configs.save_config"):
                result = create_training_config("round_dir", "test_model", volumes)
    assert isinstance(result, dict)


def test_create_prediction_configs(mock_click):
    mock_click.prompt.return_value = 30000
    volumes = [{"zarr_container": "test.zarr", "raw_dataset": "raw"}]
    train_config = {"setup_dir": "setup_dir", "max_iterations": 30001}
    with patch("bootstrapper.run.configs.open"):
        with patch("json.load", return_value={"outputs": ["output"]}):
            with patch("bootstrapper.run.configs.check_and_update", return_value={}):
                with patch("bootstrapper.run.configs.save_config"):
                    with patch(
                        "bootstrapper.run.configs.get_roi",
                        return_value=([0, 0, 0], [100, 100, 100], [1, 1, 1]),
                    ):
                        result = create_prediction_configs(volumes, train_config)
    assert isinstance(result, tuple)


def test_create_segmentation_configs(mock_click):
    mock_click.prompt.side_effect = [False, False, 10]
    volumes = [{"zarr_container": "test.zarr", "raw_mask_dataset": None}]
    with patch("bootstrapper.run.configs.get_rag_db_config", return_value={}):
        with patch("bootstrapper.run.configs.check_and_update", return_value={}):
            with patch("bootstrapper.run.configs.save_config"):
                result = create_segmentation_configs(volumes, "affs_ds", "setup_dir")
    assert isinstance(result, str)


def test_create_evaluation_configs(mock_click):
    mock_click.prompt.return_value = True
    volumes = [{"zarr_container": "test.zarr"}]
    with patch("bootstrapper.run.configs.check_and_update", return_value={}):
        with patch("bootstrapper.run.configs.save_config"):
            result = create_evaluation_configs(
                volumes, "segs", True, ["pred_3d_lsds"], "setup_dir"
            )
    assert isinstance(result, str)


def test_create_filter_configs(mock_click):
    volumes = [
        {
            "zarr_container": "test.zarr",
            "raw_dataset": "raw",
            "raw_mask_dataset": None,
            "voxel_size": [1, 1, 1],
        }
    ]
    with patch("bootstrapper.run.configs.check_and_update", return_value={}):
        with patch("bootstrapper.run.configs.save_config"):
            result = create_filter_configs(
                volumes, "segs", "eval_dir", "round_1", "model", "setup_dir"
            )
    assert isinstance(result, list)


def test_make_round_configs():
    with patch("bootstrapper.run.configs.create_training_config", return_value={}):
        with patch(
            "bootstrapper.run.configs.create_prediction_configs",
            return_value=("affs", ["pred"], True, "setup_dir"),
        ):
            with patch(
                "bootstrapper.run.configs.create_segmentation_configs",
                return_value="segs",
            ):
                with patch(
                    "bootstrapper.run.configs.create_evaluation_configs",
                    return_value="eval_dir",
                ):
                    with patch(
                        "bootstrapper.run.configs.create_filter_configs",
                        return_value=[],
                    ):
                        result = make_round_configs("round_dir", "model", [])
    assert isinstance(result, list)
    rmtree("round_dir")


def test_get_roi(mock_click):
    mock_click.prompt.side_effect = ["0 0 0", "10 10 10"]
    mock_open_ds = MagicMock()
    mock_open_ds.roi = MagicMock()
    mock_open_ds.roi.shape = [10, 10, 10]
    mock_open_ds.voxel_size = [1, 1, 1]
    mock_open_ds.axis_names = ["z", "y", "x"]
    with patch("bootstrapper.run.configs.open_ds", return_value=mock_open_ds):
        result = get_roi("test.zarr")
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(hasattr(x, "__iter__") for x in result)


def test_make_configs(mock_click):
    mock_click.prompt.side_effect = [1, "round_1"]
    with patch("os.listdir", return_value=[]):
        with patch("bootstrapper.run.configs.make_volumes", return_value=[]):
            with patch("bootstrapper.run.configs.choose_model", return_value="model"):
                with patch(
                    "bootstrapper.run.configs.make_round_configs", return_value=[]
                ):
                    make_configs("base_dir")
    rmtree("base_dir")


if __name__ == "__main__":
    pytest.main()
