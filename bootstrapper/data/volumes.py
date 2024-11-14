import os
import click
import zarr
import subprocess

from ..styles import cli_echo, cli_prompt, cli_confirm


def process_zarr(path, output_zarr, type, style="prepare"):

    cli_echo(f"Processing {path}", style)
    in_array = zarr.open(path)

    do_bbox = cli_confirm(
        "Perform bounding box crop?", style, default=False if type == "raw" else True
    )
    copy_to_output = cli_confirm(
        f"Copy {path} to output container {output_zarr}?", style, default=False
    )

    if do_bbox or copy_to_output:
        out_ds_path = cli_prompt(
            f"Enter output {type.upper()} dataset path",
            style,
            default=os.path.join(output_zarr, type),
        )
    else:
        out_ds_path = path

    if do_bbox:
        subprocess.run(
            [
                "bs",
                "utils",
                "bbox",
                "-i",
                path,
                "-o",
                out_ds_path,
            ]
        )
    else:
        if not copy_to_output:
            return out_ds_path, in_array.attrs["voxel_size"]

        # copy contents of in_array into a new zarr array at out_ds_path, with attrs
        out_array = zarr.open(
            out_ds_path,
            mode="w",
            shape=in_array.shape,
            chunks=in_array.chunks,
            dtype=in_array.dtype,
        )
        out_array.attrs.update(in_array.attrs)
        out_array[:] = in_array[:]

    return out_ds_path, in_array.attrs["voxel_size"]


def process_non_zarr(path, output_zarr, type, style="prepare"):

    dataset_name = cli_prompt(
        f"Enter output {type.upper()} dataset path",
        style,
        default=f"{type}",
    )
    out_array = os.path.join(output_zarr, dataset_name)
    dtype = cli_prompt(
        "Enter data type",
        style,
        default="uint32" if type == "labels" else "uint8",
    )
    voxel_size = tuple(
        cli_prompt(
            "Enter voxel size (space separated integers)",
            style,
            default="1 1 1",
        ).split()
    )
    voxel_offset = tuple(
        cli_prompt(
            "Enter voxel offset (space separated integers)",
            style,
            default="0 0 0",
        ).split()
    )
    axis_names = tuple(
        cli_prompt(
            click.style("Enter axis names (space separated strings)", style),
            default="z y x",
        ).split()
    )
    units = tuple(
        cli_prompt(
            click.style("Enter units (space separated strings)", style),
            default="nm nm nm",
        ).split()
    )

    crop = cli_confirm(
        click.style("Perform bounding box crop?", style),
        default=False if type == "raw" else True,
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


def process_dataset(path, output_zarr, type, style="prepare"):

    if path is None:
        return None, None, None

    if os.path.isdir(path) and os.path.exists(os.path.join(path, ".zarray")):
        ds_name, vs = process_zarr(path, output_zarr, type)
    elif os.path.isdir(path) and path.endswith(".zarr") or path.endswith(".zarr/"):
        raise ValueError(
            f"{path} is not a valid zarr dataset, it must contain a .zarray file"
        )
    else:
        ds_name, vs = process_non_zarr(path, output_zarr, type)

    out_ds_name = f"{ds_name}"

    # make or provide mask
    if cli_confirm(
        f"Make or provide {type} mask?",
        style,
        default=False,
    ):
        if cli_confirm("Make mask?", style, default=False):
            mask_ds_name = out_ds_name.replace(type, f"{type}_mask")
            subprocess.run(
                [
                    "bs",
                    "utils",
                    "mask",
                    "-i",
                    out_ds_name,
                    "-o",
                    mask_ds_name,
                    "-m",
                    type,
                ]
            )
        elif cli_confirm("Provide mask?", style, default=False):
            mask_ds_name = cli_prompt(
                "Enter path to mask dataset", style, type=click.Path(exists=True)
            )
    else:
        mask_ds_name = None

    return out_ds_name, mask_ds_name, vs


def prepare_volume(volume_path, style="prepare"):
    # check if volume path ends in .zarr or .zarr/, else raise
    if volume_path.endswith(".zarr") or volume_path.endswith(".zarr/"):
        output_zarr = os.path.abspath(volume_path)
    else:
        raise ValueError(f"Volume (output container) path must end in .zarr")

    # get volume name
    volume_name = os.path.basename(volume_path).split(".zarr")[0]

    # procress raw
    while True:
        try:
            path = cli_prompt(
                f"Enter path to input RAW 3D image, directory of 2D images, or zarr array for {volume_name}",
                style,
                type=click.Path(exists=True),
            )
            path = os.path.abspath(path)
            raw_ds, raw_mask, raw_vs = process_dataset(path, output_zarr, "raw")
            break
        except click.Abort:
            raise
        except Exception as e:
            cli_echo(f"{e}, try again.", style)

    # process labels
    click.echo()
    path = cli_prompt(
        f"Enter path to input LABELS 3D image, directory of 2D images, or zarr container for {volume_name} (enter to skip)",
        style,
        default=" ",
        show_default=False,
    )
    path = None if path == " " else os.path.abspath(path)
    obj_ds, obj_mask, obj_vs = process_dataset(path, output_zarr, "labels")

    if obj_ds is None or raw_vs != obj_vs:
        vs = raw_vs if raw_vs is not None else obj_vs
    else:
        vs = raw_vs

    return {
        "name": volume_name,
        "output_container": output_zarr,
        "raw_dataset": os.path.abspath(raw_ds),
        "raw_mask_dataset": None if raw_mask is None else os.path.abspath(raw_mask),
        "labels_dataset": None if obj_ds is None else os.path.abspath(obj_ds),
        "labels_mask_dataset": None if obj_mask is None else os.path.abspath(obj_mask),
        "voxel_size": list(vs),
    }
