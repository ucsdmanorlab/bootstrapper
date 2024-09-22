import unittest
import click
from click.testing import CliRunner
from bootstrapper.utils.mask import make_mask
import tempfile
import zarr
import numpy as np
import daisy

class TestMask(unittest.TestCase):

    def setUp(self):
        # Create temporary zarr arrays for testing
        self.temp_dir = tempfile.mkdtemp()
        self.input_array = zarr.open(self.temp_dir + '/input.zarr', mode='w', shape=(20, 20, 20), dtype=np.uint8)
        self.input_array[5:15, 5:15, 5:15] = 1
        self.input_path = self.temp_dir + '/input.zarr'
        self.args = ['-i', self.input_path]
        
        # Test cases
        self.arg_combos = [
            self.args + ['-m', 'raw'],
            self.args + ['-m', 'labels'],
            self.args + ['-m', 'raw', '-o', self.temp_dir + '/output_raw.zarr'],
            self.args + ['-m', 'labels', '-o', self.temp_dir + '/output_labels.zarr'],
        ]

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)

    def check_function(self, result, args):
        mode = args[args.index('-m') + 1]
        output_path = str(result).strip().split("Writing mask to ")[-1].split("\n")[0]
        output_array = zarr.open(output_path, mode='r')
        self.assertEqual(output_array.shape, (20, 20, 20))
        self.assertEqual(output_array.dtype, np.uint8)
        self.assertEqual(np.array_equal(output_array, self.input_array), True)

    def test_make_mask_cli(self):
        runner = CliRunner()
        for args in self.arg_combos:
            result = runner.invoke(make_mask, args)
            self.assertEqual(result.exit_code, 0)
            self.assertIn('Writing mask to', result.output)
            self.check_function(result.output, args)

    def test_make_mask_function(self):
        for args in self.arg_combos:
            in_array = args[1]
            mode = args[args.index('-m') + 1]
            out_array = args[args.index('-o') + 1] if '-o' in args else None

            with click.Context(make_mask) as ctx:
                result = ctx.invoke(make_mask, in_array=in_array, out_array=out_array, mode=mode)

            self.check_function(result, args)

if __name__ == '__main__':
    unittest.main()
