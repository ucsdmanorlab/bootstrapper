import os
from scipy.ndimage import find_objects
from funlib.persistence import open_ds, prepare_ds
import click

@click.command()
@click.option('--in_array', '-i', type=click.Path(exists=True), required=True, help='Path to input Zarr array.', prompt='Enter the path to the input array')
@click.option('--out_array', '-o', default=None, help='Path to output array')
@click.option('--padding', '-p', type=int, default=0, help='Padding to add to the bounding box.')
def bbox(in_array, out_array, padding):
    """Crop an array to its bounding box."""

    in_ds = open_ds(in_array)
    arr = in_ds[in_ds.roi]

    slices = find_objects(arr > 0)[0]
    slices = [slice(max(0, s.start - padding), min(s.stop + padding, arr.shape[i])) for i, s in enumerate(slices)]
    slices = tuple(slices)

    new_offset = [in_ds.offset[i]+(slices[i].start * in_ds.voxel_size[i]) for i in range(3)]

    if out_array is None:
        out_array = os.path.join(os.path.dirname(in_array), os.path.basename(in_array) + '_bbox')
        click.echo(f"Output array not specified.")

    click.echo(f"Writing to {out_array}")
    cropped_arr = arr[slices]

    out_ds = prepare_ds(
        out_array,
        shape=cropped_arr.shape,
        offset=new_offset,
        voxel_size=in_ds.voxel_size,
        axis_names=in_ds.axis_names,
        units=in_ds.units,
        dtype=in_ds.dtype,
        chunk_shape=in_ds.chunk_shape,
        mode='w'
    )
    out_ds[out_ds.roi] = cropped_arr

    click.echo(f"Done!")
    return out_array

if __name__ == '__main__':
    bbox()