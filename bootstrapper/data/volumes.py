import os
import click
import zarr
import subprocess

DEFAULT_PROMPT_STYLE = {"fg": "white"}
DEFAULT_INFO_STYLE = {"fg": "white", "bold": True}
DEFAULT_PROMPT_SUFFIX = click.style(" >>> ", **DEFAULT_PROMPT_STYLE)


def process_zarr(path, output_zarr, type):

    click.secho(f"Processing {path} to {output_zarr}", **DEFAULT_INFO_STYLE)
    in_f = zarr.open_group(path)
    click.secho(in_f.tree())

    in_ds_name = click.prompt(
        click.style(
            f"Enter input {type.upper()} dataset name contained in the Zarr container",
            **DEFAULT_PROMPT_STYLE,
        ),
        type=str,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
    )
    if in_ds_name is None:
        return None, None

    out_ds_name = click.prompt(
        click.style("Enter output dataset name", **DEFAULT_PROMPT_STYLE),
        default=f"{type}",
        type=str,
        show_default=True,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
    )

    if click.confirm(
        click.style("Perform bounding box crop?", **DEFAULT_PROMPT_STYLE),
        default=False if type == "raw" else True,
        show_default=True,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
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
            and click.confirm(
                click.style(
                    f"Rename {in_ds_name} to {out_ds_name}?", **DEFAULT_PROMPT_STYLE
                ),
                default=True,
                prompt_suffix=DEFAULT_PROMPT_SUFFIX,
            )
        ):
            click.secho(f"Renaming {in_ds_name} to {out_ds_name}")
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
            click.secho(
                f"Copying {path}/{in_ds_name} to {output_zarr}/{out_ds_name}",
                **DEFAULT_INFO_STYLE,
            )
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
        click.style("Enter output dataset name", **DEFAULT_PROMPT_STYLE),
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
    if path.endswith(".zarr") or path.endswith(".zarr/"):
        ds_name, vs = process_zarr(path, output_zarr, type)
    else:
        ds_name, vs = process_non_zarr(path, output_zarr, type)

    if ds_name is None:
        return None, None, None

    out_ds_name = f"{ds_name}"

    if click.confirm(
        click.style(f"Make {type} mask?", **DEFAULT_PROMPT_STYLE),
        default=False,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
    ):
        mask_ds_name = out_ds_name.replace(type, f"{type}_mask")
        subprocess.run(
            ["bs", "utils", "mask", "-i", out_ds_name, "-o", mask_ds_name, "-m", type]
        )
    else:
        mask_ds_name = None

    return out_ds_name, mask_ds_name, vs


def prepare_volume(volume_path):
    # check if volume path ends in .zarr or .zarr/, else raise
    if volume_path.endswith(".zarr") or volume_path.endswith(".zarr/"):
        output_zarr = os.path.abspath(volume_path)
    else:
        raise ValueError(f"Volume path must end in .zarr")

    # procress raw
    path = click.prompt(
        click.style(
            f"Enter path to input RAW tif directory, tif stack, or zarr container for volume {volume_path}",
            **DEFAULT_PROMPT_STYLE,
        ),
        type=click.Path(exists=True),
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
    )
    path = os.path.abspath(path)
    raw_ds, raw_mask, raw_vs = process_dataset(path, output_zarr, "raw")

    # process labels
    path = click.prompt(
        click.style(
            f"Enter path to input LABELS tif directory, tif stack, or zarr container for volume {volume_path}",
            **DEFAULT_PROMPT_STYLE,
        ),
        type=click.Path(exists=True),
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
    )
    path = os.path.abspath(path)
    obj_ds, obj_mask, obj_vs = process_dataset(path, output_zarr, "labels")

    if raw_vs != obj_vs:
        click.secho(
            f"Voxel sizes do not match! {raw_vs} vs {obj_vs}", fg="red", bold=True
        )
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
