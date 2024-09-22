import unittest
import click
from click.testing import CliRunner
from bootstrapper.utils.size_filter import size_filter
import tempfile
import zarr
import numpy as np

class TestSizeFilter(unittest.TestCase):

    def setUp(self):
        # Create temporary zarr arrays for testing
        self.temp_dir = tempfile.mkdtemp()
        self.input_array = zarr.open(self.temp_dir + '/input.zarr', mode='w', shape=(30, 30, 30), dtype=np.uint32)
        self.input_array[0:1, 0:9, 0:10] = 1
        self.input_array[15:20, 15:20, 15:20] = 2
        self.input_array[25:26, 25:26, 25:26] = 3
        self.input_path = self.temp_dir + '/input.zarr'
        self.args = ['-i', self.input_path]
        
        # Test cases
        self.arg_combos = [
            self.args + ['-s', '100'],
            self.args + ['-o', self.temp_dir + '/output.zarr', '-s', '100'],
        ]

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)

    def check_function(self, result, args):
        output_path = str(result).strip().split("Filtered output at ")[-1].split("\n")[0]
        output_array = zarr.open(output_path, mode='r')
        self.assertEqual(output_array.shape, (30, 30, 30))
        self.assertEqual(output_array.dtype, np.uint32)
        unique_labels = np.unique(output_array)
        self.assertLess(len(unique_labels), 3)  # At least one label should be filtered out

    def test_size_filter_cli(self):
        runner = CliRunner()
        for args in self.arg_combos:
            result = runner.invoke(size_filter, args)
            self.assertEqual(result.exit_code, 0)
            self.assertIn('Filtered output at', result.output)
            self.check_function(result.output, args)

    def test_size_filter_function(self):
        for args in self.arg_combos:
            in_array = args[1]
            out_array = args[args.index('-o') + 1] if '-o' in args else None
            size_threshold = int(args[args.index('-s') + 1]) if '-s' in args else 500

            with click.Context(size_filter) as ctx:
                result = ctx.invoke(size_filter, in_array=in_array, out_array=out_array, size_threshold=size_threshold)

            self.check_function(result, args)

if __name__ == '__main__':
    unittest.main()
