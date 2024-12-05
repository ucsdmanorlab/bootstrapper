import click
import toml
import logging
import time
import os
from pprint import pprint
from funlib.geometry import Coordinate
from funlib.persistence import open_ds
from volara.blockwise import AffAgglom
from volara.datasets import Affs, Labels
from volara.dbs import SQLite, PostgreSQL

logging.getLogger().setLevel(logging.INFO)


def agglomerate(config, frags_ds_name=None):

    # Extract arguments from config
    affs_dataset = config["affs_dataset"]  # Name of affinities dataset
    fragments_dataset_prefix = config["fragments_dataset"]  # Name of fragments dataset
    db_config = config["db"]  # Database configuration
    neighborhood = config["aff_neighborhood"]

    # Optional parameters
    blockwise = config.get("blockwise", False)
    num_workers = config.get("num_workers", 1)
    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)
    block_shape = config.get("block_shape", None)
    context = config.get("context", None)

    sigma = config.get("sigma", None)
    noise_eps = config.get("noise_eps", None)
    bias = config.get("bias", None)

    if sigma is not None or noise_eps is not None or bias is not None:
        if frags_ds_name is None:
            shift_name = []
            if noise_eps is not None:
                shift_name.append(f"{noise_eps}")
            if sigma is not None:
                shift_name.append(f"{"_".join([str(x) for x in sigma])}")
            if bias is not None:
                shift_name.append(f"{"_".join([str(x) for x in bias])}")
            shift_name = "--".join(shift_name)
            frags_ds_name = os.path.join(fragments_dataset_prefix, shift_name)
        else:
            shift_name = os.path.basename(frags_ds_name)

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

    affinities = Affs(
        store=affs_dataset,
        neighborhood=neighborhood
    )
    frags = open_ds(frags_ds_name, 'r')
    fragments = Labels(store=frags_ds_name)
    # print(fragments.axis_names, fragments.units)

    # get total ROI
    if roi_offset is not None:
        roi = (roi_offset, roi_shape)
    else:
        roi = (frags.roi.offset, frags.roi.shape)

    # get block size, context
    if blockwise:
        if block_shape is not None:
            block_size = Coordinate(block_shape)
        else:
            block_size = Coordinate(frags.chunk_shape)

        if context is not None:
            context = Coordinate(context)
        else:
            context = Coordinate(
                [2,] * frags.roi.dims
            )

    else:  # blockwise is False
        block_size = frags.shape
        context = Coordinate(
            [0,] * frags.roi.dims
        )
        num_workers = 1

    # Affinity Agglomeration across blocks
    aff_agglom = AffAgglom(
        db=db,
        affs_data=affinities,
        frags_data=fragments,
        block_size=block_size,
        roi=roi,
        context=context,
        num_workers=num_workers,
        scores={"zyx_aff": affinities.neighborhood},
    )
    aff_agglom.run_blockwise(multiprocessing=True)


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def agglom(config_file):
    """
    Agglomerate fragments using waterz and daisy.
    """

    # Load config file
    with open(config_file, "r") as f:
        toml_config = toml.load(f)

    config = toml_config | toml_config["mws_params"]
    for x in config.copy():
        if x.endswith("_params"):
            del config[x]

    pprint(config)

    start = time.time()
    agglomerate(config)
    end = time.time()

    seconds = end - start
    logging.info(f"Total time to agglomerate mutex fragments: {seconds} ")

if __name__ == "__main__":
    agglom()