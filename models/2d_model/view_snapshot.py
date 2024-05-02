import neuroglancer
import numpy as np
import os
import sys
import zarr

# set bind address if accessing remotely
neuroglancer.set_server_bind_address('localhost',bind_port=3334)

# path to snapshots/batch_{i}.zarr after training for i iterations
f = zarr.open(sys.argv[1])

# get all datasets saved in snapshot
datasets = [i for i in os.listdir(sys.argv[1]) if "." not in i]

res = f[datasets[0]].attrs["resolution"]

viewer = neuroglancer.Viewer()

# allows us to treat batches as a spatial dimension and cycle through them. c^
# tells us to view channels as a non spatial dimension, and renders with a
# shader
dims = neuroglancer.CoordinateSpace(
    names=["b", "c^", "y", "x"], units="nm", scales=res + res
)

with viewer.txn() as s:
    for ds in datasets:
        print(ds)
        offset = f[ds].attrs["offset"]

        # add a dummy batch and channel dim
        offset = [
            0,
        ] * 2 + [int(i / j) for i, j in zip(offset, res)]

        # load data to numpy array
        data = f[ds][:]

        shader = """
void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(2)))
        );
}"""

        # get data (np.array for example)
        # create neuroglancer local volume
        # add local volume as a layer (imagelayer, segmentationlayer)

        try:
            s.layers[ds] = neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(
                    data=data, voxel_offset=offset, dimensions=dims
                ),
                shader=shader,
            )

        except Exception as e:
            print(ds, e)

    # s.layout = "yz"

print(viewer)
