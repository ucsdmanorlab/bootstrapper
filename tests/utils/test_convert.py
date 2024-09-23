import unittest
import click
from click.testing import CliRunner
from bootstrapper.utils.convert import convert
import tempfile
import zarr
from funlib.persistence import open_ds
import numpy as np
import os
import shutil

class TestConvert(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_tif = os.path.join(self.temp_dir, 'input.tif')
        np.random.seed(42)
        test_image = np.random.randint(0, 256, size=(10, 10, 10), dtype=np.uint8)
        from tifffile import imsave
        imsave(self.input_tif, test_image)
        self.input_path = self.input_tif
        self.output_path = os.path.join(self.temp_dir, 'output.zarr')
        self.args = ['-i', self.input_path, '-o', self.output_path]        
        # Different argument combinations
        self.arg_combos = [
            self.args,
            self.args + ['-d', 'float32'],
            self.args + ['-vs', 2, 2, 2],
            self.args + ['-vo', 1, 1, 1],
            self.args + ['-ax', 'a', 'b', 'c'],
            self.args + ['-u', 'um', 'um', 'um'],
            self.args + ['--crop'],
            self.args + ['-d', 'uint8', '-vs', 2, 4, 4, '-vo', 3, 0, 0, '-ax', 'a', 'b', 'c', '-u', '', '', '', '--crop'],
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def check_output(self, result, args):
        # get output array from result
        if type(result) == tuple:
            out_arr = result[0]
        else:
            out_arr = str(result).strip().split('Writing ')[1].strip()
        out_ds = open_ds(out_arr)

        dtype = args[args.index('-d')+1] if '-d' in args else 'uint8'
        voxel_size = args[args.index('-vs')+1:args.index('-vs')+4] if '-vs' in args else (1, 1, 1)
        voxel_offset = args[args.index('-vo')+1:args.index('-vo')+4] if '-vo' in args else (0, 0, 0)
        axis_names = args[args.index('-ax')+1:args.index('-ax')+4] if '-ax' in args else ['z', 'y', 'x']
        units = args[args.index('-u')+1:args.index('-u')+4] if '-u' in args else ['nm', 'nm', 'nm']

        # check if output array is correct
        self.assertEqual(out_ds.shape, (10, 10, 10))
        self.assertEqual(out_ds.dtype, np.dtype(dtype))
        self.assertEqual(out_ds.voxel_size, tuple(voxel_size))
        self.assertEqual(out_ds.offset, tuple(vo * vs for vo, vs in zip(voxel_offset, voxel_size)))
        self.assertEqual(out_ds.axis_names, axis_names)
        self.assertEqual(out_ds.units, units)

    def test_convert_cli(self):
        runner = CliRunner()
        for args in self.arg_combos:
            result = runner.invoke(convert, args)
            self.assertEqual(result.exit_code, 0)
            self.assertIn('Writing ', result.output)
            self.check_output(result.output, args)
            shutil.rmtree(self.output_path)

    def test_convert_function(self):
        for args in self.arg_combos:
            in_path = args[1]
            out_array = args[3]
            dtype = args[args.index('-d')+1] if '-d' in args else 'uint8'
            voxel_size = tuple(map(int, args[args.index('-vs')+1:args.index('-vs')+4])) if '-vs' in args else (1, 1, 1)
            voxel_offset = tuple(map(int, args[args.index('-vo')+1:args.index('-vo')+4])) if '-vo' in args else (0, 0, 0)
            axis_names = tuple(args[args.index('-ax')+1:args.index('-ax')+4]) if '-ax' in args else ('z', 'y', 'x')
            units = tuple(args[args.index('-u')+1:args.index('-u')+4]) if '-u' in args else ('nm', 'nm', 'nm')
            crop = '--crop' in args

            with click.Context(convert) as ctx:
                result = ctx.invoke(convert, in_path=in_path, out_array=out_array,
                                    dtype=dtype, voxel_size=voxel_size, voxel_offset=voxel_offset,
                                    axis_names=axis_names, units=units, crop=crop)
            self.check_output(result, args)
            shutil.rmtree(self.output_path)

if __name__ == '__main__':
    unittest.main()
