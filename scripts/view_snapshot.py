import neuroglancer
import numpy as np
import os
import sys
import zarr


def create_coordinate_space(voxel_size, is_3d):
    names = ['c^', 'z', 'y', 'x'] if is_3d else ['b', 'c^', 'y', 'x']
    scales = [1,] + voxel_size if is_3d else [1, 1] + voxel_size[-2:]
    return neuroglancer.CoordinateSpace(names=names, units='nm', scales=scales)

def process_dataset(f, ds, is_3d):
    data = f[ds][:]
    vs = f[ds].attrs['voxel_size']
    offset = f[ds].attrs['offset']

    if not is_3d:
        if ds != 'raw' and len(data.shape) == 5:
            data = np.squeeze(data, axis=-3)
            offset = offset[1:]
            vs = vs[1:]
        elif ds == 'raw' and len(data.shape) == 4 and len(vs) == 3:
            vs = vs[1:]

    offset = [0, 0] + [int(i / j) for i, j in zip(offset, vs)]
    return data, vs, offset

def create_shader(shape):
    rgb =  """
    void main() {
        emitRGB(
            vec3(
                toNormalized(getDataValue(0)),
                toNormalized(getDataValue(1)),
                toNormalized(getDataValue(2))
            )
        );
    }
    """

    rg = """
    void main() {
        emitRGB(
            vec3(
                toNormalized(getDataValue(0)),
                toNormalized(getDataValue(1)),
                toNormalized(getDataValue())
            )
        );
    }
    """
    if len(shape) == 5:
        return rgb if shape[1] >= 3 else rg
    elif len(shape) == 4:
        return rgb if shape[0] == 3 else rg
    else:
        return None

def add_layer(s, ds, data, offset, dims, is_segmentation=False):
    if is_segmentation:
        layer_class = neuroglancer.SegmentationLayer
    else:
        layer_class = neuroglancer.ImageLayer

    s.layers[ds] = layer_class(
        source=neuroglancer.LocalVolume(
            data=data,
            voxel_offset=offset,
            dimensions=dims
        ),
        shader=create_shader(data.shape) if not is_segmentation else None
    )

def main(zarr_path):
    neuroglancer.set_server_bind_address('0.0.0.0')
    viewer = neuroglancer.Viewer()

    f = zarr.open(zarr_path)
    datasets = [i for i in os.listdir(zarr_path) if '.' not in i]
    
    # Determine if the data is 3D based on the first dataset
    is_3d = len(f[datasets[0]].shape) == 5

    dims = create_coordinate_space(f[datasets[0]].attrs['voxel_size'], is_3d)

    with viewer.txn() as s:
        for ds in datasets:
            try:
                data, _, offset = process_dataset(f, ds, is_3d)
                is_segmentation = 'label' in ds or 'seg' in ds
                add_layer(s, ds, data, offset, dims, is_segmentation)
                print(f"Added layer: {ds}")
            except Exception as e:
                print(f"Error processing dataset {ds}: {e}")

        s.layout = 'yx'

    print(viewer)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -i view_snapshot.py <path_to_snapshot_zarr>")
        sys.exit(1)
    main(sys.argv[1])