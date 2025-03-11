import os
import time
import logging
import toml
import click
from pprint import pprint
from pathlib import Path

from volara.datasets import Labels
from volara.dbs import SQLite, PostgreSQL
from volara.lut import LUT
from volara.blockwise import GraphMWS

logging.getLogger().setLevel(logging.INFO)


def global_mws(config, frags_ds_name=None):

    fragments_dataset_prefix = config["fragments_dataset"]  # Name of fragments dataset
    lut_dir = config["lut_dir"]
    os.makedirs(lut_dir, exist_ok=True)
    db_config = config["db"]  # Database configuration

    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)

    sigma = config.get("sigma", None)
    noise_eps = config.get("noise_eps", None)
    bias = config.get("bias", None)

    global_bias = tuple(config.get("global_bias", [1.0,-0.5]))

    if sigma is not None or noise_eps is not None or bias is not None:
        if frags_ds_name is None:
            shift_name = []
            if noise_eps is not None:
                shift_name.append(f"{noise_eps}")
            if sigma is not None:
                shift_name.append(f"{"_".join([str(x) for x in sigma])}")
            if bias is not None:
                shift_name.append(f"b{"_".join([str(x) for x in bias])}")
            shift_name = "--".join(shift_name)
            frags_ds_name = os.path.join(fragments_dataset_prefix, shift_name)
            lut_name = os.path.join(lut_dir, shift_name)
        else:
            shift_name = os.path.basename(frags_ds_name)
            lut_name = os.path.join(lut_dir, shift_name)

    fragments = Labels(store=frags_ds_name)
    fragments_roi = fragments.array().roi

    # get total ROI
    if roi_offset is not None:
        roi = (roi_offset, roi_shape)
    else:
        roi = (fragments_roi.offset, fragments_roi.shape)

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

    lut = LUT(path=lut_name)

    global_mws = GraphMWS(
        db=db,
        lut=lut,
        roi=roi,
        weights={"zyx_aff": global_bias},
    )
    global_mws.run_blockwise(multiprocessing=False)


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def luts(config_file):
    """
    Find connected components of region graph and store lookup tables.
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
    global_mws(config)
    end = time.time()

    seconds = end - start
    logging.info(f"Total time to do global mutex watershed: {seconds}")

if __name__ == "__main__":
    luts()
