import unittest
import click
from click.testing import CliRunner
from bootstrapper.utils.outlier_filter import outlier_filter
import tempfile
import zarr
import numpy as np


class TestOutlierFilter(unittest.TestCase):

    def setUp(self):
        # Create temporary zarr arrays for testing
        self.temp_dir = tempfile.mkdtemp()
        self.input_array = zarr.open(
            self.temp_dir + "/input.zarr", mode="w", shape=(30, 30, 30), dtype=np.uint32
        )
        self.input_array[0:6, 0:15, 0:15] = 1
        self.input_array[15:16, 15:25, 15:25] = 2
        self.input_array[26:27, 26:29, 26:29] = 3
        self.input_array[0:1, 0:1, 0:1] = 4
        self.input_path = self.temp_dir + "/input.zarr"
        self.args = ["-i", self.input_path]

        # Test cases
        self.arg_combos = [
            self.args + ["-s", "1.0"],
            self.args + ["-o", self.temp_dir + "/output.zarr", "-s", "1.0"],
        ]

    def tearDown(self):
        # Clean up temporary files
        import shutil

        shutil.rmtree(self.temp_dir)

    def check_function(self, result, args):
        output_path = (
            str(result).strip().split("Filtered output at ")[-1].split("\n")[0]
        )
        output_array = zarr.open(output_path, mode="r")
        self.assertEqual(output_array.shape, (30, 30, 30))
        self.assertEqual(output_array.dtype, np.uint32)
        unique_labels = np.unique(output_array)
        self.assertLess(
            len(unique_labels), 5
        )  # At least one label should be filtered out

    def test_outlier_filter_cli(self):
        runner = CliRunner()
        for args in self.arg_combos:
            result = runner.invoke(outlier_filter, args)
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Filtered output at", result.output)
            self.check_function(result.output, args)

    def test_outlier_filter_function(self):
        for args in self.arg_combos:
            in_labels = args[1]
            out_labels = args[args.index("-o") + 1] if "-o" in args else None
            sigma = float(args[args.index("-s") + 1]) if "-s" in args else 3.0

            with click.Context(outlier_filter) as ctx:
                result = ctx.invoke(
                    outlier_filter,
                    in_labels=in_labels,
                    out_labels=out_labels,
                    sigma=sigma,
                )

            self.check_function(result, args)


if __name__ == "__main__":
    unittest.main()
