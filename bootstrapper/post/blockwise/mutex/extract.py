import time
import logging
import toml
import click
from pprint import pprint
import os
from pathlib import Path

import zarr
from funlib.persistence import open_ds
from volara.datasets import Labels
from volara.lut import LUT
from volara.blockwise import Relabel

logging.getLogger().setLevel(logging.INFO)


def extract_segmentation(config, frags_ds_name=None):
    fragments_dataset_prefix = config["fragments_dataset"]
    lut_dir = config["lut_dir"]
    seg_dataset_prefix = config["seg_dataset_prefix"]

    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)
    block_size = config.get("block_shape", None)
    num_workers = config.get("num_workers", 40)

    filter_fragments = config.get("filter_fragments", None)
    sigma = config.get("sigma", None)
    noise_eps = config.get("noise_eps", None)
    bias = config.get("bias", None)
    strides = config.get("strides", None)
    global_bias = config.get("global_bias", -0.5)

    if any([sigma, noise_eps, bias, filter_fragments, strides]):
        if frags_ds_name is None:
            shift_name = []
            if filter_fragments is not None:
                shift_name.append(f"filt{filter_fragments}")
            if noise_eps is not None:
                shift_name.append(f"eps{noise_eps}")
            if sigma is not None:
                shift_name.append(f"sigma{'_'.join([str(x) for x in sigma])}")
            if bias is not None:
                shift_name.append(f"bias{'_'.join([str(x) for x in bias])}")
            if strides is not None:
                shift_name.append(f"strides{'_'.join([str(x[0]) for x in strides])}")
            shift_name = "--".join(shift_name)
            frags_ds_name = os.path.join(fragments_dataset_prefix, shift_name)
            lut_name = os.path.join(lut_dir, shift_name)
        else:
            shift_name = os.path.basename(frags_ds_name)
            lut_name = os.path.join(lut_dir, shift_name)

    frags = open_ds(frags_ds_name, "r")
    fragments = Labels(store=frags_ds_name)
    # get total ROI
    if roi_offset is not None:
        roi = (roi_offset, roi_shape)
    else:
        roi = (frags.roi.offset, frags.roi.shape)

    if block_size is None:
        block_size = frags.chunk_shape

    seg_name = os.path.join(seg_dataset_prefix, f"gb{str(global_bias)}--{shift_name}")
    segments = Labels(store=seg_name)

    lut = LUT(path=lut_name)

    # Extract segments
    relabel_task = Relabel(
        frags_data=fragments,
        seg_data=segments,
        lut=lut,
        roi=roi,
        block_size=block_size,
        num_workers=num_workers
    )
    relabel_task.run_blockwise(multiprocessing=True)

    # temp fix: update zarr attributes
    zattrs = {
        'axis_names': frags.axis_names,
        'units': frags.units,
    }
    segments_zarr = zarr.open(seg_name, mode="a")

    for k,v in zattrs.items():
        segments_zarr.attrs[k] = v



@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option(
    "--thresholds",
    "-t",
    help="Thresholds to extract segmentations from (space separated floats)",
)
def extract(config_file, thresholds=None):
    """
    Extracts segmentations from fragments using LUTs
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
    extract_segmentation(config)
    end = time.time()

    seconds = end - start
    logging.info(f"Total time to extract_segmentation: {seconds}")

if __name__ == "__main__":
    extract()
