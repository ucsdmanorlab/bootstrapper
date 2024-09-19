import neuroglancer
import numpy as np
import os
import sys
import zarr


def create_coordinate_space(voxel_size, is_2d):
    names = ['c^', 'z', 'y', 'x'] if not is_2d else ['b', 'c^', 'y', 'x']
    scales = [1,] + voxel_size if not is_2d else voxel_size[-2:] + voxel_size[-2:]
    return neuroglancer.CoordinateSpace(names=names, units='nm', scales=scales)

def process_dataset(f, ds, is_2d):
    data = f[ds][:]
    vs = f[ds].attrs['voxel_size']
    offset = f[ds].attrs['offset']

    if is_2d:
        if ds != 'raw' and len(data.shape) == 5:
            data = np.squeeze(data, axis=-3)
            offset = offset[1:]
            vs = vs[1:]
        elif ds == 'raw' and len(data.shape) == 4 and len(vs) == 3:
            vs = vs[1:]

    if is_2d:
        offset = [0, 0] + [int(i / j) for i, j in zip(offset, vs)]
    else:
        offset = [0,] + [int(i / j) for i, j in zip(offset, vs)]

    return data, vs, offset

def create_shader(ds, is_2d):
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

    if is_2d:
        if ds == 'raw':
            shader = rgb
        else:
            shader = rg
    else:
        shader = rgb
    return shader


def main(zarr_path):
    neuroglancer.set_server_bind_address('0.0.0.0')
    viewer = neuroglancer.Viewer()

    f = zarr.open(zarr_path)
    datasets = [i for i in os.listdir(zarr_path) if '.' not in i]
    
    # Determine if the data is 3D based on the first dataset
    try:
        raw_shape = f["raw"].shape
    except KeyError:
        raw_shape = f[datasets[0]].shape
    shape = f[datasets[0]].shape
    print(raw_shape, shape)
    is_2d = (len(shape) == 5 and shape[-3] == 1) and (len(raw_shape) == 4)

    dims = create_coordinate_space(f[datasets[0]].attrs['voxel_size'], is_2d)

    with viewer.txn() as s:
        for ds in datasets:
            try:
                data, _, offset = process_dataset(f, ds, is_2d)
                is_segmentation = 'label' in ds or 'seg' in ds

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
                )
                if not is_segmentation:
                    s.layers[ds].shader = create_shader(ds, is_2d)

                print(f"Added layer: {ds}")
            except Exception as e:
                print(f"Error processing dataset {ds}: {e}")

        s.layout = "yz"

    print(viewer)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please supply the path to the snapshot zarr.")
        sys.exit(1)
    main(sys.argv[1])