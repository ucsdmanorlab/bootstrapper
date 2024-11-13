import os
import click
import zarr
import subprocess

DEFAULT_PROMPT_STYLE = {"fg": "white"}
DEFAULT_INFO_STYLE = {"fg": "white", "bold": True}
DEFAULT_PROMPT_SUFFIX = click.style(" >>> ", **DEFAULT_PROMPT_STYLE)


def process_zarr(path, output_zarr, type):

    click.secho(f"Processing {path}", **DEFAULT_INFO_STYLE)
    in_array = zarr.open(path)

    do_bbox = click.confirm(
        click.style("Perform bounding box crop?", **DEFAULT_PROMPT_STYLE),
        default=False if type == "raw" else True,
        show_default=True,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
    )
    copy_to_output = click.confirm(
        click.style(f"Copy {path} to output container {output_zarr}?", **DEFAULT_PROMPT_STYLE),
        default=False,
        show_default=True,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
    )

    if do_bbox or copy_to_output:
        out_ds_path = click.prompt(
            click.style(f"Enter output {type.upper()} dataset path", **DEFAULT_PROMPT_STYLE),
            default=os.path.join(output_zarr, type),
            type=str,
            show_default=True,
            prompt_suffix=DEFAULT_PROMPT_SUFFIX)
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
        out_array = zarr.open(out_ds_path, mode='w', shape=in_array.shape, chunks=in_array.chunks, dtype=in_array.dtype)
        out_array.attrs.update(in_array.attrs)
        out_array[:] = in_array[:]

    return out_ds_path, in_array.attrs["voxel_size"]


def process_non_zarr(path, output_zarr, type):

    dataset_name = click.prompt(
        click.style(f"Enter output {type.upper()} dataset path", **DEFAULT_PROMPT_STYLE),
        default=f"{type}",
        type=str,
        show_default=True,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
    )
    out_array = os.path.join(output_zarr, dataset_name)
    dtype = click.prompt(
        click.style("Enter data type", **DEFAULT_PROMPT_STYLE),
        default="uint32" if type == "labels" else "uint8",
        type=str,
        show_default=True,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
    )
    voxel_size = tuple(
        click.prompt(
            click.style(
                "Enter voxel size (space separated integers)", **DEFAULT_PROMPT_STYLE
            ),
            default="1 1 1",
            type=str,
            show_default=True,
            prompt_suffix=DEFAULT_PROMPT_SUFFIX,
        ).split()
    )
    voxel_offset = tuple(
        click.prompt(
            click.style(
                "Enter voxel offset (space separated integers)", **DEFAULT_PROMPT_STYLE
            ),
            default="0 0 0",
            type=str,
            show_default=True,
            prompt_suffix=DEFAULT_PROMPT_SUFFIX,
        ).split()
    )
    axis_names = tuple(
        click.prompt(
            click.style(
                "Enter axis names (space separated strings)", **DEFAULT_PROMPT_STYLE
            ),
            default="z y x",
            type=str,
            show_default=True,
            prompt_suffix=DEFAULT_PROMPT_SUFFIX,
        ).split()
    )
    units = tuple(
        click.prompt(
            click.style(
                "Enter units (space separated strings)", **DEFAULT_PROMPT_STYLE
            ),
            default="nm nm nm",
            type=str,
            show_default=True,
            prompt_suffix=DEFAULT_PROMPT_SUFFIX,
        ).split()
    )

    crop = click.confirm(
        click.style("Perform bounding box crop?", **DEFAULT_PROMPT_STYLE),
        default=False if type == "raw" else True,
        show_default=True,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
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

    if path is None:
        return None, None, None

    if os.path.isdir(path) and os.path.exists(os.path.join(path, ".zarray")):
        ds_name, vs = process_zarr(path, output_zarr, type)
    elif os.path.isdir(path) and path.endswith(".zarr") or path.endswith(".zarr/"):
        raise ValueError(f"{path} is not a valid zarr dataset, it must contain a .zarray file")
    else:
        ds_name, vs = process_non_zarr(path, output_zarr, type)

    out_ds_name = f"{ds_name}"

    # make or provide mask
    if click.confirm(
        click.style(f"Make or provide {type} mask?", **DEFAULT_PROMPT_STYLE),
        default=False,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
    ):
        if click.confirm(
            click.style("Make mask?", **DEFAULT_PROMPT_STYLE),
            default=False,
            prompt_suffix=DEFAULT_PROMPT_SUFFIX,
        ):
            mask_ds_name = out_ds_name.replace(type, f"{type}_mask")
            subprocess.run(
                ["bs", "utils", "mask", "-i", out_ds_name, "-o", mask_ds_name, "-m", type]
            )
        elif click.confirm(
            click.style("Provide mask?", **DEFAULT_PROMPT_STYLE),
            default=False,
            prompt_suffix=DEFAULT_PROMPT_SUFFIX,
        ):
            mask_ds_name = click.prompt(
                click.style(
                    "Enter path to mask dataset", **DEFAULT_PROMPT_STYLE
                ),
                type=click.Path(exists=True),
                prompt_suffix=DEFAULT_PROMPT_SUFFIX,
            )
    else:
        mask_ds_name = None

    return out_ds_name, mask_ds_name, vs


def prepare_volume(volume_path):
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
            path = click.prompt(
                click.style(
                    f"Enter path to input RAW 3D image, directory of 2D images, or zarr array for {volume_name}",
                    **DEFAULT_PROMPT_STYLE,
                ),
                type=click.Path(exists=True),
                prompt_suffix=DEFAULT_PROMPT_SUFFIX,
            )
            path = os.path.abspath(path)
            raw_ds, raw_mask, raw_vs = process_dataset(path, output_zarr, "raw")
            break
        except Exception as e:
            click.secho(f"{e}, try again.", **DEFAULT_INFO_STYLE)

    # process labels
    path = click.prompt(
        click.style(
            f"Enter path to input LABELS 3D image, directory of 2D images, or zarr container for {volume_name} (enter to skip)",
            **DEFAULT_PROMPT_STYLE,
        ),
        default=" ",
        show_default=False,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
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
