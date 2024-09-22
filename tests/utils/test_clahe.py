import unittest
import click
from click.testing import CliRunner
from bootstrapper.utils.clahe import clahe
import tempfile
import zarr
import numpy as np
import shutil

class TestClahe(unittest.TestCase):

    def setUp(self):
        # Create a temporary zarr array for testing
        self.temp_dir = tempfile.mkdtemp()
        self.input_array = zarr.open(self.temp_dir + '/input.zarr', mode='w', shape=(1, 10, 10), dtype=np.uint8)
        self.input_array[:] = np.random.randint(0, 256, size=(1, 10, 10), dtype=np.uint8)
        self.input_path = self.temp_dir + '/input.zarr'
        self.args = ['-i', self.input_path]
        
        # Different argument combinations
        self.arg_combos = [
            self.args,
            self.args + ['-o', self.temp_dir + '/output.zarr'],
            self.args + ['-b', '1', '10', '10'],
            self.args + ['-c', '0', '2', '2'],
            self.args + ['-o', self.temp_dir + '/output.zarr', '-b', '1', '10', '10', '-c', '0', '2', '2'],
        ]

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)

    def check_function(self, result, args):
        output_path = str(result).strip().split('Output created at ')[-1].strip().split("\n")[0]
        output_array = zarr.open(output_path, mode='r')
        self.assertEqual(output_array.shape, self.input_array.shape)
        self.assertNotEqual(np.array_equal(output_array[:], self.input_array[:]), True)
        shutil.rmtree(output_path)

    def test_clahe_cli(self):
        runner = CliRunner()
        for args in self.arg_combos:
            result = runner.invoke(clahe, args)
            self.assertEqual(result.exit_code, 0)
            self.assertIn('Output created at', result.output)
            self.check_function(result.output, args)

    def test_clahe_function(self):
        for args in self.arg_combos:
            in_array = args[1]
            out_array = self.temp_dir + '/output.zarr' if '-o' in args else None
            block_shape = tuple(map(int, args[args.index('-b')+1:args.index('-b')+4])) if '-b' in args else (1, 256, 256)
            context = tuple(map(int, args[args.index('-c')+1:args.index('-c')+4])) if '-c' in args else (0, 128, 128)

            with click.Context(clahe) as ctx:
                result = ctx.invoke(clahe, in_arr=in_array, out_arr=out_array, block_shape=block_shape, context=context)

            self.check_function(result, args)

if __name__ == '__main__':
    unittest.main()
