import os
import click
import toml
import time
import logging
import zarr
from funlib.geometry import Coordinate
from funlib.persistence import open_ds
from volara.blockwise import ExtractFrags
from volara.datasets import Affs, Labels, Raw
from volara.dbs import SQLite, PostgreSQL

logging.getLogger().setLevel(logging.INFO)


def extract_fragments(config):
    affs_dataset = config["affs_dataset"]  # Name of affinities dataset
    fragments_dataset_prefix = config["fragments_dataset"]  # Name of fragments dataset
    db_config = config["db"]  # Database configuration
    mask_dataset = config.get("mask_dataset", None)

    # required mws params
    neighborhood = config.get("aff_neighborhood", None)
    bias = config.get("bias", None)

    # optional mws params
    filter_fragments = config.get(
        "filter_fragments", None
    )  # Filter fragments with average affinity lower than this value
    sigma = config.get("sigma", None)
    noise_eps = config.get("noise_eps", None)
    strides = config.get("strides", None)
    randomized_strides = config.get("randomized_strides", False)
    remove_debris = config.get("remove_debris", 0)

    # Optional parameters
    roi_offset = config.get("roi_offset", None)  # Offset of ROI
    roi_shape = config.get("roi_shape", None)  # Shape of ROI
    blockwise = config.get("blockwise", False)  # Perform blockwise extraction
    num_workers = config.get("num_workers", 1)  # Number of workers to use
    block_shape = config.get("block_shape", None)  # Shape of block
    context = config.get("context", None)  # Context for block

    # load affs
    affs = open_ds(affs_dataset)

    # validate neighborhood and bias
    if neighborhood is None:
        raise ValueError("Affinities neighborrhood must be provided")
    if bias is None:
        raise ValueError("Affinities bias must be provided")

    # assert (
    #     len(neighborhood) == affs.shape[0]
    # ), "Number of offsets must match number of affinities channels"
    assert len(neighborhood) == len(
        bias
    ), "Numbes of biases must match number of affinities channels"

    shift_name = []
    if sigma is not None or noise_eps is not None or bias is not None:
        if noise_eps is not None:
            shift_name.append(f"{noise_eps}")
        if sigma is not None:
            shift_name.append(f"{"_".join([str(x) for x in sigma])}")
        if bias is not None:
            shift_name.append(f"b{"_".join([str(x) for x in bias])}")

    shift_name = "--".join(shift_name)

    # get total ROI
    if roi_offset is not None:
        roi = (roi_offset, roi_shape)
    else:
        roi = (affs.roi.offset, affs.roi.shape)

    # get block size, context
    if blockwise:
        if block_shape is not None:
            block_size = Coordinate(block_shape)
        else:
            block_size = Coordinate(affs.chunk_shape[1:])

        if context is not None:
            context = Coordinate(context)
        else:
            context = Coordinate(
                [2,] * affs.roi.dims
            )

    else:  # blockwise is False
        block_size = affs.shape[1:]
        context = Coordinate(
            [0,] * affs.roi.dims
        )
        num_workers = 1

    # prepare fragments array
    frags_ds_name = os.path.join(fragments_dataset_prefix, shift_name)

    # prepare RAG provider
    logging.info(msg="Creating RAG DB...")
    # Configure your db
    if "db_file" in db_config:
        db = SQLite(
            path=db_config["db_file"],
            edge_attrs={
                "zyx_aff": "float",
            }
        )
    else:
        db = PostgreSQL(
            name=db_config["db_name"],
            host=db_config["db_host"],
            user=db_config["db_user"],
            password=db_config["db_password"],
            edge_attrs={
                "zyx_aff": "float",
            }
        )

    # Configure your arrays
    affinities = Affs(
        store=affs_dataset,
        neighborhood=neighborhood,
    )
    fragments = Labels(store=frags_ds_name)

    if mask_dataset is not None:
        mask_dataset = Raw(store=mask_dataset)

    # Extract Fragments
    extract_frags = ExtractFrags(
        db=db,
        affs_data=affinities,
        frags_data=fragments,
        mask_data=mask_dataset,
        block_size=block_size,
        context=context,
        num_workers=num_workers,
        roi=roi,
        bias=bias,
        sigma=sigma,
        noise_eps=noise_eps,
        filter_fragments=filter_fragments,
        remove_debris=remove_debris,
        strides=strides,
        randomized_strides=randomized_strides
    )

    extract_frags.drop() # to restart + remove logs cache
    extract_frags.run_blockwise(multiprocessing=blockwise)

    # tempfix: preserve zattrs in frags
    zattrs = {
        'axis_names': affs.axis_names[1:],
        'units': affs.units,
    }
    frags_zarr = zarr.open(frags_ds_name, mode="r+")

    for k,v in zattrs.items():
        frags_zarr.attrs[k] = v


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def frags(config_file):
    """
    Extract fragments from affinities using daisy.
    """

    # Load config file
    with open(config_file, "r") as f:
        toml_config = toml.load(f)

    config = toml_config | toml_config["mws_params"]
    for x in config.copy():
        if x.endswith("_params"):
            del config[x]

    start = time.time()
    extract_fragments(config)
    end = time.time()

    seconds = end - start
    logging.info(f"Total time to extract mutex fragments: {seconds} ")

if __name__ == "__main__":
    frags()