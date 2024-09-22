import unittest
import click
from click.testing import CliRunner
from bootstrapper.utils.bbox import bbox
import tempfile
import zarr
import numpy as np

class TestBbox(unittest.TestCase):

    def setUp(self):
        # Create a temporary zarr array for testing
        self.temp_dir = tempfile.mkdtemp()
        self.input_array = zarr.open(self.temp_dir + '/input.zarr', mode='w', shape=(10, 10, 10), dtype=np.uint8)
        self.input_array[2:8, 2:8, 2:8] = 1
        self.input_path = self.temp_dir + '/input.zarr'
        self.args = ['-i', self.input_path]
        
        # Edge use cases
        self.arg_combos = [
            self.args,
            self.args + ['-o', self.temp_dir + '/output.zarr'],
            self.args + ['--padding', '1'],
            self.args + ['-o', self.temp_dir + '/output.zarr', '--padding', '1'],
        ]

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)

    def check_function(self, result, args):
        padding = int(args[args.index('--padding') + 1]) if '--padding' in args else 0
        output_path = str(result).strip().split("Writing to ")[-1].split("\n")[0]
        output_array = zarr.open(output_path, mode='r')
        expected_shape = (6 + 2 * padding,) * 3
        self.assertEqual(output_array.shape, expected_shape)

    def test_bbox_cli(self):
        runner = CliRunner()
        for args in self.arg_combos:
            result = runner.invoke(bbox, args)
            self.assertEqual(result.exit_code, 0)
            self.assertIn('Writing to', result.output)
            self.check_function(result.output, args)

    def test_bbox_function(self):
        for args in self.arg_combos:
            in_array = args[1]
            out_array = self.temp_dir + '/output.zarr' if '-o' in args else None
            padding = int(args[args.index('--padding') + 1]) if '--padding' in args else 0

            with click.Context(bbox) as ctx:
                result = ctx.invoke(bbox, in_array=in_array, out_array=out_array, padding=padding)

            self.check_function(result, args)

if __name__ == '__main__':
    unittest.main()
