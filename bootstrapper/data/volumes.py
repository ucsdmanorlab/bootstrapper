import os
import click
import zarr
import subprocess
import logging

# Set up logging
logger = logging.getLogger(__name__)


def process_zarr(path, output_zarr, type):

    logger.info(f"Processing {path} to {output_zarr}")
    in_f = zarr.open(path)
    logger.info(in_f.tree())

    in_ds_name = click.prompt(
        f"Enter input {type.upper()} dataset name contained in the Zarr container",
        type=str,
    )
    if in_ds_name is None:
        return None, None

    out_ds_name = click.prompt(
        "Enter output dataset name",
        default=f"volumes/{type}",
        type=str,
        show_default=True,
    )

    if click.confirm(
        "Perform bounding box crop?",
        default=False if type == "raw" else True,
        show_default=True,
    ):
        subprocess.run(
            [
                "bs",
                "utils",
                "bbox",
                "-i",
                os.path.join(path, in_ds_name),
                "-o",
                os.path.join(output_zarr, out_ds_name),
            ]
        )

    else:
        out_f = zarr.open(output_zarr, "a")

        if (
            os.path.abspath(path) == os.path.abspath(output_zarr)
            and in_ds_name != out_ds_name
            and click.confirm(f"Rename {in_ds_name} to {out_ds_name}?", default=True)
        ):
            logger.info(f"Renaming {in_ds_name} to {out_ds_name}")
            in_f.store.rename(in_ds_name, in_ds_name + "__tmp")
            # in_f.create_group('/'.join(out_ds_name.split('/')[:-1]))
            in_f.store.rename(in_ds_name + "__tmp", out_ds_name)
            in_ds_name = out_ds_name
        elif (
            os.path.abspath(path) == os.path.abspath(output_zarr)
            and in_ds_name == out_ds_name
        ):
            pass
        else:
            logger.info(f"Copying {path}/{in_ds_name} to {output_zarr}/{out_ds_name}")
            out_f[out_ds_name] = in_f[in_ds_name]
            out_f[out_ds_name].attrs["offset"] = in_f[in_ds_name].attrs["offset"]
            out_f[out_ds_name].attrs["voxel_size"] = in_f[in_ds_name].attrs[
                "voxel_size"
            ]
            out_f[out_ds_name].attrs["axis_names"] = in_f[in_ds_name].attrs[
                "axis_names"
            ]
            out_f[out_ds_name].attrs["units"] = in_f[in_ds_name].attrs["units"]

    return os.path.join(output_zarr, out_ds_name), in_f[in_ds_name].attrs["voxel_size"]


def process_non_zarr(path, output_zarr, type):

    dataset_name = click.prompt(
        "Enter output dataset name",
        default=f"volumes/{type}",
        type=str,
        show_default=True,
    )
    out_array = os.path.join(output_zarr, dataset_name)
    dtype = click.prompt(
        "Enter data type",
        default="uint32" if type == "labels" else "uint8",
        type=str,
        show_default=True,
    )
    voxel_size = tuple(
        click.prompt(
            "Enter voxel size (space separated integers)",
            default="1 1 1",
            type=str,
            show_default=True,
        ).split()
    )
    voxel_offset = tuple(
        click.prompt(
            "Enter voxel offset (space separated integers)",
            default="0 0 0",
            type=str,
            show_default=True,
        ).split()
    )
    axis_names = tuple(
        click.prompt(
            "Enter axis names (space separated strings)",
            default="z y x",
            type=str,
            show_default=True,
        ).split()
    )
    units = tuple(
        click.prompt(
            "Enter units (space separated strings)",
            default="nm nm nm",
            type=str,
            show_default=True,
        ).split()
    )
    crop = click.confirm(
        "Perform bounding box crop?",
        default=False if type == "raw" else True,
        show_default=True,
    )

    args = [
        "bs",
        "utils",
        "convert",
        "-i",
        path,
        "-o",
        out_array,
        "-d",
        dtype,
        "-vs",
        *voxel_size,
        "-vo",
        *voxel_offset,
        "-ax",
        *axis_names,
        "-u",
        *units,
    ]
    if crop:
        args.append("-c")

    subprocess.run(args)

    return out_array, tuple(map(int, voxel_size))


def process_dataset(path, output_zarr, type):
    if path.endswith(".zarr") or path.endswith(".zarr/"):
        ds_name, vs = process_zarr(path, output_zarr, type)
    else:
        ds_name, vs = process_non_zarr(path, output_zarr, type)

    if ds_name is None:
        return None, None, None

    out_ds_name = f"{ds_name}"
    if type == "raw":
        if click.confirm("Apply CLAHE?", default=True):
            out_ds_name += "_clahe"
            subprocess.run(["bs", "utils", "clahe", "-i", ds_name, "-o", out_ds_name])

    if click.confirm("Generate scale pyramid?", default=False):
        in_file = output_zarr
        in_ds_name = out_ds_name.split(".zarr/")[-1]
        scales = tuple(
            click.prompt(
                "Enter scales for each axis (space separated integers)",
                default="1 2 2",
                type=str,
                show_default=True,
            ).split()
        )
        chunk_shape = tuple(
            click.prompt(
                "Enter chunk shape (space separated integers)",
                default="8 256 256",
                type=str,
                show_default=True,
            ).split()
        )
        mode = click.prompt(
            "Enter mode",
            type=click.Choice(["up", "down"]),
            show_default=True,
            default="down",
        )

        subprocess.run(
            [
                "bs",
                "utils",
                "scale-pyramid",
                "-f",
                in_file,
                "-ds",
                in_ds_name,
                "-s",
                *scales,
                "-c",
                *chunk_shape,
                "-m",
                mode,
            ]
        )
        out_ds_name = os.path.join(out_ds_name, "s0")

    if click.confirm(f"Make {type} mask?", default=False):
        mask_ds_name = out_ds_name.replace(type, f"{type}_mask")
        subprocess.run(
            ["bs", "utils", "mask", "-i", out_ds_name, "-o", mask_ds_name, "-m", type]
        )
    else:
        mask_ds_name = None

    return out_ds_name, mask_ds_name, vs


def prepare_volume(base_dir, volume_index):
    output_zarr = click.prompt(
        "Enter path to output zarr container",
        default=os.path.join(base_dir, f"volume_{volume_index + 1}.zarr"),
        type=click.Path(),
    )
    output_zarr = os.path.abspath(output_zarr)

    path = click.prompt(
        f"Enter path to input RAW tif directory, tif stack, or zarr container for volume {volume_index + 1}",
        type=click.Path(exists=True),
    )
    path = os.path.abspath(path)

    raw_ds, raw_mask, raw_vs = process_dataset(path, output_zarr, "raw")

    path = click.prompt(
        f"Enter path to input LABELS tif directory, tif stack, or zarr container for volume {volume_index + 1}",
        type=click.Path(exists=True),
    )
    path = os.path.abspath(path)
    obj_ds, obj_mask, obj_vs = process_dataset(path, output_zarr, "labels")

    if raw_vs != obj_vs:
        logger.warning(f"Voxel sizes do not match: {raw_vs} vs {obj_vs}")
        vs = raw_vs if raw_vs is not None else obj_vs
    else:
        vs = raw_vs

    return {
        "zarr_container": os.path.abspath(output_zarr),
        "raw_dataset": None if raw_ds is None else os.path.abspath(raw_ds),
        "raw_mask_dataset": None if raw_mask is None else os.path.abspath(raw_mask),
        "labels_dataset": None if obj_ds is None else os.path.abspath(obj_ds),
        "labels_mask_dataset": None if obj_mask is None else os.path.abspath(obj_mask),
        "voxel_size": list(vs),
    }


def make_volumes(base_dir, num_volumes=None):
    """Prepare volumes for bootstrapping."""

    logger.info(f"Processing volumes in {base_dir}")

    if not base_dir:
        base_dir = click.prompt(
            "Enter the base directory path",
            default=".",
            type=click.Path(file_okay=False, dir_okay=True),
        )
        os.makedirs(base_dir, exist_ok=True)

    if not num_volumes:
        num_volumes = click.prompt(
            "How many volumes for this round?", default=1, type=int
        )

    volumes = []
    for i in range(num_volumes):
        logger.info(f"Processing volume {i+1}")
        volume_info = prepare_volume(base_dir, i)
        if volume_info:
            volumes.append(volume_info)

    logger.info(f"Processed volumes: {volumes}")
    return volumes
