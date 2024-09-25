import click
import neuroglancer
import numpy as np
import os
import sys
import zarr
import subprocess
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--snapshot",
    "-s",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to the Zarr container of a snapshot",
)
@click.argument("datasets", nargs=-1)
def view(snapshot, datasets):
    """
    View a snapshot or run neuroglancer -d <args>

    Parameters
    ----------
    snapshot : str
        Path to the Zarr container of a snapshot.
    datasets : str
        Datasets to be viewed with neuroglancer.

    Returns
    -------
    None
    """
    logger.info("Starting view command")
    if snapshot:
        if snapshot.endswith(".zarr") or snapshot.endswith(".zarr/"):
            logger.info(f"Viewing snapshot: {snapshot}")
            view_snapshot(snapshot)
            click.pause("Press any key to exit...")
        else:
            logger.error("Snapshot path must end with '.zarr'")
            click.echo("Error: Snapshot path must end with '.zarr'", err=True)
            sys.exit(1)
    else:
        logger.info(f"Running neuroglancer with datasets: {datasets}")
        neuroglancer_args = ["neuroglancer", "-d"] + list(datasets)
        subprocess.run(neuroglancer_args)


def create_coordinate_space(voxel_size, is_2d):
    names = ["c^", "z", "y", "x"] if not is_2d else ["b", "c^", "y", "x"]
    scales = (
        [
            1,
        ]
        + voxel_size
        if not is_2d
        else voxel_size[-2:] + voxel_size[-2:]
    )
    logger.debug(f"Creating coordinate space with names: {names}, scales: {scales}")
    return neuroglancer.CoordinateSpace(names=names, units="nm", scales=scales)


def process_dataset(f, ds, is_2d):
    data = f[ds][:]
    vs = f[ds].attrs["voxel_size"]
    offset = f[ds].attrs["offset"]

    if is_2d:
        if ds != "raw" and len(data.shape) == 5:
            data = np.squeeze(data, axis=-3)
            offset = offset[1:]
            vs = vs[1:]
        elif ds == "raw" and len(data.shape) == 4 and len(vs) == 3:
            vs = vs[1:]

    if is_2d:
        offset = [0, 0] + [int(i / j) for i, j in zip(offset, vs)]
    else:
        offset = [
            0,
        ] + [int(i / j) for i, j in zip(offset, vs)]

    logger.debug(
        f"Processed {ds}: shape={data.shape}, voxel_size={vs}, offset={offset}"
    )
    return data, vs, offset


def create_shader(ds, is_2d):
    rgb = """
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
        if ds == "raw":
            shader = rgb
        else:
            shader = rg
    else:
        shader = rgb
    logger.debug(f"Created shader for dataset: {ds}")
    return shader


def view_snapshot(zarr_path):
    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()

    f = zarr.open(zarr_path)
    datasets = [i for i in os.listdir(zarr_path) if "." not in i]

    # Determine if the data is 3D based on the first dataset
    try:
        raw_shape = f["raw"].shape
    except KeyError:
        raw_shape = f[datasets[0]].shape
    shape = f[datasets[0]].shape
    logger.info(f"Raw shape: {raw_shape}, pred shape: {shape}")
    is_2d = (len(shape) == 5 and shape[-3] == 1) and (len(raw_shape) == 4)

    dims = create_coordinate_space(f[datasets[0]].attrs["voxel_size"], is_2d)

    with viewer.txn() as s:
        for ds in datasets:
            try:
                data, _, offset = process_dataset(f, ds, is_2d)
                is_segmentation = "label" in ds or "seg" in ds

                if is_segmentation:
                    layer_class = neuroglancer.SegmentationLayer
                else:
                    layer_class = neuroglancer.ImageLayer

                s.layers[ds] = layer_class(
                    source=neuroglancer.LocalVolume(
                        data=data, voxel_offset=offset, dimensions=dims
                    ),
                )
                if not is_segmentation:
                    s.layers[ds].shader = create_shader(ds, is_2d)

                logger.info(f"Added layer: {ds}")
            except Exception as e:
                logger.error(f"Error processing dataset {ds}: {e}")

        s.layout = "yz"

    logger.info("Viewer setup complete")
    print(viewer)
