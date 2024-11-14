import daisy
import click
import toml
import logging
import numpy as np
import os
import time
import zarr
from functools import partial
from pprint import pprint

from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, prepare_ds


logging.getLogger().setLevel(logging.INFO)


def segment_in_block(fragments, segmentation, lut, block):
    import numpy as np
    from funlib.segment.arrays import replace_values

    # load fragments
    fragments = fragments[block.read_roi]

    # replace values, write to empty array
    relabelled = np.zeros_like(fragments)
    relabelled = replace_values(
        in_array=fragments, old_values=lut[0], new_values=lut[1], out_array=relabelled
    )

    # write to segmentation array
    segmentation[block.write_roi] = relabelled


def extract_segmentations(config):
    # read config
    fragments_dataset = config["fragments_dataset"]
    lut_dir = config["lut_dir"]
    seg_dataset_prefix = config["seg_dataset_prefix"]
    thresholds = config["thresholds"]
    merge_function = config["merge_function"]
    num_workers = config["num_workers"]
    pprint(config)

    # load fragments
    fragments = open_ds(fragments_dataset)
    voxel_size = fragments.voxel_size

    # get LUT dir
    lut_dir = os.path.join(lut_dir, "fragment_segment")

    # get ROIs
    if "block_shape" in config and config["block_shape"] not in [None, "roi"]:
        block_size = Coordinate(config["block_shape"]) * voxel_size
    else:
        block_size = Coordinate(fragments.chunk_shape) * voxel_size

    if "roi_offset" in config and "roi_shape" in config:
        roi_offset = config["roi_offset"]
        roi_shape = config["roi_shape"]
    else:
        roi_offset = None
        roi_shape = None

    if roi_offset is not None:
        total_roi = Roi(roi_offset, roi_shape)
    else:
        total_roi = fragments.roi

    read_roi = write_roi = Roi((0,) * fragments.roi.dims, block_size)

    for threshold in thresholds:
        seg_name: str = f"{seg_dataset_prefix}/{merge_function}_{str(threshold)}"

        start: float = time.time()
        logging.info(f"Writing {seg_name}")

        segmentation = prepare_ds(
            store=seg_name,
            shape=total_roi.shape / voxel_size,
            offset=total_roi.offset,
            voxel_size=voxel_size,
            axis_names=fragments.axis_names,
            units=fragments.units,
            chunk_shape=write_roi.shape / voxel_size,
            compressor=zarr.get_codec({"id": "blosc"}),
            dtype=np.uint64,
            mode="w",
        )

        # read LUT
        lut_filename = (
            f"{config["db"]['edges_table']}_{str(int(threshold*100)).zfill(2)}"
        )
        lut = os.path.join(lut_dir, lut_filename + ".npz")
        assert os.path.exists(path=lut), f"{lut} does not exist"
        lut = np.load(file=lut)["fragment_segment_lut"]

        logging.info(f"Found {len(lut[0])} fragments in LUT")
        num_segments: int = len(np.unique(ar=lut[1]))
        logging.info(f"Relabelling fragments to {num_segments} segments")

        task = daisy.Task(
            task_id="ExtractSegmentsTask",
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=partial(segment_in_block, fragments, segmentation, lut),
            fit="shrink",
            num_workers=num_workers,
        )

        done: bool = daisy.run_blockwise(tasks=[task])

        if not done:
            raise RuntimeError(
                "Extraction of segmentation from LUT failed for (at least) one block"
            )

    return True


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def extract(config_file, **kwargs):
    """
    Extracts segmentations from fragments using LUTs
    """

    with open(config_file, "r") as f:
        config = toml.load(f)

    start = time.time()
    extract_segmentations(config)
    end = time.time()

    seconds = end - start
    logging.info(f"Total time to extract_segmentation: {seconds}")
