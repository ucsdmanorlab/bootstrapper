import os
import click
import zarr
from funlib.persistence import open_ds, prepare_ds

#from .data import bbox, scale_pyr, make_3d_zarr_array, mask_blockwise

def process_zarr(path, zarr_file):
    with zarr.open(path, 'r') as f:
        click.echo(f.tree())

    in_ds = click.prompt("Enter input dataset name contained in the Zarr container", default="volumes/raw")
    dataset_name = click.prompt("Enter output dataset name", default=in_ds)

    if click.confirm("Perform bounding box crop?", default=False):
        bbox(path, in_ds, zarr_file, dataset_name)
    else:
        in_f = zarr.open(path)
        out_f = zarr.open(zarr_file, "a")
        if os.path.abspath(path) == os.path.abspath(zarr_file) and in_ds != dataset_name:
            click.echo(f"Renaming {in_ds} to {dataset_name}..")
            in_f.store.rename(in_ds, in_ds + '__tmp')
            in_f.create_group('/'.join(dataset_name.split('/')[:-1]))
            in_f.store.rename(in_ds + '__tmp', dataset_name)
        elif os.path.abspath(path) == os.path.abspath(zarr_file) and in_ds == dataset_name:
            pass
        else:
            click.echo(f"Copying {path}/{in_ds} to {zarr_file}/{dataset_name}..")
            out_f[dataset_name] = in_f[in_ds]
            out_f[dataset_name].attrs['offset'] = in_f[in_ds].attrs['offset']
            out_f[dataset_name].attrs['voxel_size'] = in_f[in_ds].attrs['voxel_size']

    return dataset_name

def process_non_zarr(path, zarr_file):
    voxel_size = [
        click.prompt("Enter Z voxel size (in world units)", default="1"),
        click.prompt("Enter YX voxel size (in world units)", default="1")
    ]

    dataset_name = click.prompt("Enter output dataset name", default="volumes/raw/s0")
    make_3d_zarr_array(path, zarr_file, dataset_name, *voxel_size)

    return dataset_name

def prepare_volume(base_dir, volume_index):
    path = click.prompt(f"Enter path to input tif directory, tif stack, or zarr container for volume {volume_index + 1}", type=click.Path(exists=True))
    if not path:
        return None

    zarr_file = os.path.join(base_dir, f"volume_{volume_index + 1}.zarr")

    if path.endswith('.zarr'):
        dataset_name = process_zarr(path, zarr_file)
    else:
        dataset_name = process_non_zarr(path, zarr_file)

    if click.confirm("Run downscale pyramid?", default=True):
        scale_pyr(zarr_file, dataset_name)

    mask_type = 'img' if 'image' in dataset_name else 'obj'
    if click.confirm(f"Make {'raw' if mask_type == 'img' else 'object'} masks?", default=True):
        source_dataset = dataset_name
        transform = lambda s: '/'.join(parts[:-2] + [parts[-2] + '_mask'] + parts[-1:]) if (parts := s.split('/')) and len(parts) > 1 else s + '_mask'

        if click.confirm("Do masking downscaled dataset then upscale?", default=True):
            if not source_dataset.endswith('s0'):
                source_dataset += '/s0'
            source_dataset = source_dataset.replace('s0', 's2')

        out_mask_dataset = transform(source_dataset)
        mask_blockwise(zarr_file, source_dataset, out_mask_dataset, mask_type)

        if click.confirm("Upscale the mask?", default=True):
            scale_pyr(zarr_file, out_mask_dataset, mode='up')

    return {"path": path, "zarr_file": zarr_file, "dataset_name": dataset_name}

@click.command()
@click.option('--base-dir', prompt="Enter base directory", default='.', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--num-volumes', type=int)
def get_volumes(base_dir, num_volumes):
    os.makedirs(base_dir, exist_ok=True)

    if num_volumes is None:
        num_volumes = click.prompt("How many volumes for this round?", default=1, type=int)

    volumes = []
    for i in range(num_volumes):
        click.echo(f"Processing volume {i+1}:")
        volume_info = prepare_volume(base_dir, i)
        if volume_info:
            volumes.append(volume_info)

    click.echo(f"Processed volumes: {volumes}")
    return volumes