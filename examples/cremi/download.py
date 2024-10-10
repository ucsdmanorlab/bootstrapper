import subprocess
import h5py
import zarr
import tifffile
import numpy as np
from skimage.transform import rescale

# download cremi_c data
subprocess.run([
    "wget",
    "https://cremi.org/static/data/sample_C_20160501.hdf"
])

# create zarr
h5_file = h5py.File("sample_C_20160501.hdf", "r")
out_zarr = zarr.open("cremi_c.zarr","w")

out_zarr["raw"] = h5_file["volumes/raw"][:]
out_zarr["raw"].attrs["offset"] = [0,0,0]
out_zarr["raw"].attrs["voxel_size"] = list(map(int, h5_file["volumes/raw"].attrs["resolution"]))
out_zarr["raw"].attrs["axis_names"] = ["z","y","x"]
out_zarr["raw"].attrs["units"] = ["nm","nm","nm"]

out_zarr["gt_labels"] = h5_file["volumes/labels/neuron_ids"][:]

# zero out missing sections
out_zarr["gt_labels"][14,:,:] = np.zeros_like(out_zarr["gt_labels"][14,:,:])
out_zarr["gt_labels"][74,:,:] = np.zeros_like(out_zarr["gt_labels"][74,:,:])

out_zarr["gt_labels"].attrs["offset"] = [0,0,0]
out_zarr["gt_labels"].attrs["voxel_size"] = list(map(int, h5_file["volumes/labels/neuron_ids"].attrs["resolution"]))
out_zarr["gt_labels"].attrs["axis_names"] = ["z","y","x"]
out_zarr["gt_labels"].attrs["units"] = ["nm","nm","nm"]

# write sparse labels
painting = tifffile.imread("z30_painting.tif")
painting = rescale(painting, (1, 2, 2), order=0)
out_zarr["sparse_labels"] = painting
out_zarr["sparse_labels"].attrs["offset"] = [1200,0,0]
out_zarr["sparse_labels"].attrs["voxel_size"] = list(map(int, h5_file["volumes/labels/neuron_ids"].attrs["resolution"]))
out_zarr["sparse_labels"].attrs["axis_names"] = ["z","y","x"]
out_zarr["sparse_labels"].attrs["units"] = ["nm","nm","nm"]

# remote hdf5 file
h5_file.close()
subprocess.run(["rm", "sample_C_20160501.hdf"])