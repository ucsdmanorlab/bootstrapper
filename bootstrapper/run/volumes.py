import os
import click
import zarr

from bootstrapper.utils.bbox import bbox
from bootstrapper.utils.clahe import clahe
from bootstrapper.utils.scale_pyramid import scale_pyramid
from bootstrapper.utils.make_zarr import make_zarr
from bootstrapper.utils.make_mask import make_mask


def process_zarr(path, output_zarr, type):
    with zarr.open(path, 'r') as f:
        click.echo(f.tree())

    in_ds_name = click.prompt("Enter input dataset name contained in the Zarr container", type=str)
    if in_ds_name is None:
        return None, None
    
    out_ds_name = click.prompt("Enter output dataset name", default=f"volumes/{type}", type=str, show_default=True)

    if click.confirm("Perform bounding box crop?", default=False if type == 'raw' else True, show_default=True):
        with click.Context(bbox) as ctx:
            ctx.invoke(
                bbox(in_array=os.path.join(path, in_ds_name), out_array=os.path.join(output_zarr, out_ds_name))
            )
    else:
        in_f = zarr.open(path)
        out_f = zarr.open(output_zarr, "a")

        # if only the input and output zarr containers are the same, ask if to rename the dataset
        if os.path.abspath(path) == os.path.abspath(output_zarr) and in_ds_name != out_ds_name and click.confirm(f"Rename {in_ds_name} to {out_ds_name}?", default=True):
            click.echo(f"Renaming..")
            in_f.store.rename(in_ds_name, in_ds_name + '__tmp')
            in_f.create_group('/'.join(out_ds_name.split('/')[:-1]))
            in_f.store.rename(in_ds_name + '__tmp', out_ds_name)
        elif os.path.abspath(path) == os.path.abspath(output_zarr) and in_ds_name == out_ds_name: 
            pass
        else:
            click.echo(f"Copying {path}/{in_ds_name} to {output_zarr}/{out_ds_name}..")
            out_f[out_ds_name] = in_f[in_ds_name]
            out_f[out_ds_name].attrs['offset'] = in_f[in_ds_name].attrs['offset']
            out_f[out_ds_name].attrs['voxel_size'] = in_f[in_ds_name].attrs['voxel_size']
            out_f[out_ds_name].attrs['axis_names'] = in_f[in_ds_name].attrs['axis_names']
            out_f[out_ds_name].attrs['units'] = in_f[in_ds_name].attrs['units']

    return out_ds_name, in_f[in_ds_name].attrs['voxel_size']

def process_non_zarr(path, output_zarr, type):

    dataset_name = click.prompt("Enter output dataset name", default=f"volumes/{type}", type=str, show_default=True)
    out_array = os.path.join(output_zarr, dataset_name)
    dtype = click.prompt("Enter data type", default="uint32" if type == 'labels' else "uint8", type=str, show_default=True)
    # voxel_size = click.prompt()

    with click.Context(make_zarr) as ctx:
        _, voxel_size = ctx.invoke(make_zarr, in_path=path, out_array=out_array, dtype=dtype)

    return out_array, voxel_size

def process_dataset(path, output_zarr, type):
    if path.endswith('.zarr'):
        ds_name, vs = process_zarr(path, output_zarr, type)
    else:
        ds_name, vs = process_non_zarr(path, output_zarr, type)

    if ds_name is None:
        return None, None, None

    out_ds_name = f'{ds_name}'
    # apply CLAHE for image data ?
    if type == 'raw':
        if click.confirm("Apply CLAHE?", default=True):
            out_ds_name += '_clahe'
            with click.Context(clahe) as ctx:
                ctx.invoke(clahe, in_array=os.path.join(output_zarr, ds_name), out_array=os.path.join(output_zarr, out_ds_name))

    # scale pyramid ?
    if click.confirm("Generate scale pyramid?", default=False):
        with click.Context(scale_pyramid) as ctx:
            ctx.invoke(
                scale_pyramid(in_file=output_zarr, in_ds_name=out_ds_name.split('.zarr')[-1])
            )
        out_ds_name = f'{out_ds_name}/s0'

    # make mask ?
    if click.confirm(f"Make {type} mask?", default=False):
        with click.Context(make_mask) as ctx:
            mask_ds_name = ctx.invoke(make_mask, in_array=out_ds_name, mode=type)
    else:
        mask_ds_name = None

    return out_ds_name, mask_ds_name, vs

def prepare_volume(base_dir, volume_index):
    output_zarr = click.prompt("Enter path to output zarr container", default=os.path.join(base_dir, f"volume_{volume_index + 1}.zarr"))

    # prepare image array
    path = click.prompt(f"Enter path to input IMAGE tif directory, tif stack, or zarr container for volume {volume_index + 1}", type=click.Path(exists=True))
    raw_ds, raw_mask, raw_vs = process_dataset(path, output_zarr, 'raw')

    # prepare object array
    path = click.prompt(f"Enter path to input LABELS tif directory, tif stack, or zarr container for volume {volume_index + 1}", type=click.Path(exists=True))
    obj_ds, obj_mask, obj_vs = process_dataset(path, output_zarr, 'labels')

    # check voxel sizes
    if raw_vs != obj_vs:
        click.echo(f"Voxel sizes do not match: {raw_vs} vs {obj_vs}")
        # TODO: implement upscaling / downscaling to match voxel sizes
        vs = raw_vs if raw_vs is not None else obj_vs
        pass

    return {
        'zarr_container': output_zarr,
        'raw_dataset': raw_ds,
        'raw_mask_dataset': raw_mask,
        'labels_dataset': obj_ds,
        'labels_mask_dataset': obj_mask,
        'voxel_size': vs,
    }

def make_volumes(base_dir, num_volumes):
    """Prepare volumes for bootstrapping."""

    click.echo(f'{base_dir}, {num_volumes}')

    if not base_dir:
        base_dir = click.prompt("Enter the base directory path", default=".", type=click.Path(file_okay=False, dir_okay=True))
        os.makedirs(base_dir, exist_ok=True)
    
    if not num_volumes:
        num_volumes = click.prompt("How many volumes for this round?", default=1, type=int)

    volumes = []
    for i in range(num_volumes):
        click.echo(f"Processing volume {i+1}:\n")
        volume_info = prepare_volume(base_dir, i)
        if volume_info:
            volumes.append(volume_info)

    click.echo(f"Processed volumes: {volumes}")
    return volumes