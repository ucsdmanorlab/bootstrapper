import unittest
import tempfile
import zarr
import numpy as np
import os
import logging
from click.testing import CliRunner
from bootstrapper.utils.scale_pyramid import scale_pyramid

logging.basicConfig(level=logging.INFO)

class TestScalePyramid(unittest.TestCase):
    def setUp(self):
        # Create a temporary zarr array for testing
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.temp_dir, 'input.zarr')
        self.input_ds_name = 'data'
        self.input_array = zarr.open(os.path.join(self.input_path, self.input_ds_name), mode='w', shape=(64, 64, 64), dtype=np.uint8)
        self.input_array[:] = np.random.randint(0, 255, size=(64, 64, 64), dtype=np.uint8)
        self.input_array.attrs['voxel_size'] = (50, 4, 4)

    def test_scale_pyramid_downscale(self):
        runner = CliRunner()
        result = runner.invoke(scale_pyramid, [
            '--in_file', self.input_path,
            '--in_ds_name', self.input_ds_name,
            '--scales', 2, 2, 2,
            '--chunk_shape', 32, 32, 32,
            '--mode', 'down'
        ])
        logging.info(f"Downscale command output: {result.output}")
        self.assertEqual(result.exit_code, 0, f"Command failed with error: {result.exception}")
        zarr_group = zarr.open(self.input_path, mode='r')
        self.assertIn('data/s1', zarr_group, "Downscaled dataset 'data/s1' not found")
        self.assertEqual(zarr_group['data/s1'].shape, (32, 32, 32), "Incorrect shape for downscaled dataset")

    def test_scale_pyramid_upscale(self):
        runner = CliRunner()
        result = runner.invoke(scale_pyramid, [
            '--in_file', self.input_path,
            '--in_ds_name', self.input_ds_name,
            '--scales', 2, 2, 2,
            '--chunk_shape', 128, 128, 128,
            '--mode', 'up'
        ])
        logging.info(f"Upscale command output: {result.output}")
        self.assertEqual(result.exit_code, 0, f"Command failed with error: {result.exception}")
        zarr_group = zarr.open(self.input_path, mode='r')
        self.assertIn('data/s1', zarr_group, "Upscaled dataset 'data/s1' not found")
        self.assertEqual(zarr_group['data/s0'].shape, (128, 128, 128), "Incorrect shape for upscaled dataset")

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()