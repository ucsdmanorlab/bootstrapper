import os
from scipy.ndimage import find_objects
from funlib.persistence import open_ds, prepare_ds
import click


def bbox_crop(zarr_container, dataset, out_container=None, out_dataset=None):

    in_ds = open_ds(os.path.join(zarr_container, dataset))
    arr = in_ds[in_ds.roi]

    slices = find_objects(arr > 0)[0]
    new_offset = [in_ds.offset[i]+(slices[i].start * in_ds.voxel_size[i]) for i in range(3)]

    if out_container is None:
        out_container = zarr_container

    if out_dataset is None:
        out_dataset = dataset + "_cropped"

    cropped_arr = arr[slices]

    print(f"Writing {out_dataset} to {out_container}")

    out_ds = prepare_ds(
        os.path.join(out_container, out_dataset),
        shape=cropped_arr.shape,
        offset=new_offset,
        voxel_size=in_ds.voxel_size,
        axis_names=in_ds.axis_names,
        units=in_ds.units,
        dtype=in_ds.dtype,
        chunk_shape=in_ds.chunk_shape
    )
    out_ds[out_ds.roi] = cropped_arr


@click.command()
@click.argument('in_f', type=click.Path(exists=True))
@click.argument('in_ds', type=str)
@click.option('--out_f', default=None, help='Output container')
@click.option('--out_ds', default=None, help='Output dataset')
def bbox(in_f, in_ds, out_f, out_ds):
    bbox_crop(in_f, in_ds, out_f, out_ds)


if __name__ == "__main__":
    bbox()
