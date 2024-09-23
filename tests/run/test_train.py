import pytest
from click.testing import CliRunner
from unittest.mock import patch, mock_open
import yaml
import os
import tempfile
import shutil
from bootstrapper.run.train import train, run_training, extract_setup_dir

@pytest.fixture
def mock_yaml_file():
    return {
        "setup_dir": "/path/to/setup",
        "max_iterations": 1000,
        "output_dir": "/path/to/output",
        "save_checkpoints_every": 100,
        "save_snapshots_every": 200,
        "voxel_size": [2, 2, 2],
        "sigma": 20
    }

def test_extract_setup_dir(mock_yaml_file):
    with patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml_file))):
        setup_dir = extract_setup_dir("config.yaml")
        assert setup_dir == "/path/to/setup"

@patch("subprocess.run")
@patch("os.path.exists")
def test_run_training(mock_exists, mock_run, mock_yaml_file):
    mock_exists.return_value = False
    with patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml_file))):
        run_training("config.yaml")
    mock_run.assert_called_once_with(["python", "/path/to/setup/train.py", "config.yaml"])

@patch("subprocess.run")
@patch("os.path.exists")
def test_run_training_with_overrides(mock_exists, mock_run, mock_yaml_file):
    mock_exists.return_value = False
    with patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml_file))):
        run_training("config.yaml", max_iterations=2000, output_dir="/new/output")
    
    expected_config = mock_yaml_file.copy()
    expected_config["max_iterations"] = 2000
    expected_config["output_dir"] = "/new/output"
    
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args[0:2] == ["python", "/path/to/setup/train.py"]
    assert call_args[2].startswith(os.path.dirname("config.yaml"))
    assert call_args[2].endswith(".yaml")

def test_cli(mock_yaml_file):
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "output")
        yaml_file = os.path.join(temp_dir, "config.yaml")
        os.makedirs(output_dir, exist_ok=True)

        # Write the mock YAML content to a file
        with open(yaml_file, 'w') as f:
            yaml.dump(mock_yaml_file, f)

        with patch("bootstrapper.run.train.run_training") as mock_run_training:
            result = runner.invoke(train, [yaml_file, "-i", "2000", "-o", output_dir, "-v", "64 64 64", "--sigma", "640"])
            print(f"CLI Output: {result.output}")
            print(f"Exit Code: {result.exit_code}")
            assert result.exit_code == 0
            mock_run_training.assert_called_once_with(
                yaml_file,
                max_iterations=2000,
                output_dir=output_dir,
                save_checkpoints_every=None,
                save_snapshots_every=None,
                voxel_size=[64, 64, 64],
                sigma=640
            )

def test_cli_invalid_input():
    runner = CliRunner()
    result = runner.invoke(train, ["nonexistent.yaml"])
    assert result.exit_code != 0
    assert "Path 'nonexistent.yaml' does not exist" in result.output
