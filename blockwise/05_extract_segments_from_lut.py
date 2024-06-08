import daisy
import sys
import yaml
import json
import logging
import numpy as np
import os
import time
from funlib.geometry import Coordinate, Roi
from funlib.segment.arrays import replace_values
from funlib.persistence import Array, open_ds, prepare_ds


logging.getLogger().setLevel(logging.INFO)


def extract_segmentation(
        config: dict) -> bool:
    """Generate segmentation based on fragments using specified merge function.

    Args:
        fragments_file (``str``):
            Path (relative or absolute) to the zarr file where fragments are stored.

        fragments_dataset (``str``):
            The name of the fragments dataset to read from in the fragments file.

        merge_function (``str``):
            The method to use for merging fragments (e.g., 'hist_quant_75').

        thresholds (``list[float]``, optional):
            List of thresholds for segmentation. Default is [0.66, 0.68, 0.70].

        num_workers (``int``, optional):
            Number of workers to use when reading the region adjacency graph blockwise. Default is 7.

    Returns:
        ``bool``:
            True if segmentation generation was successful, False otherwise.
    """
    fragments_file = config['fragments_file']
    fragments_dataset = config['fragments_dataset']
    lut_dir = config["lut_dir"]
    seg_file = config['seg_file']
    seg_dataset = config['seg_dataset']
    thresholds = config['thresholds']
    merge_function = config['merge_function']
    num_workers = config['num_workers']

    fragments = open_ds(fragments_file, fragments_dataset)
    voxel_size = fragments.voxel_size

    if 'block_size' in config and config['block_size'] is not None:
        block_size = Coordinate(config["block_size"])
    else:
        block_size = Coordinate(fragments.chunk_shape) * voxel_size

    if 'roi_offset' in config and 'roi_shape' in config:
        roi_offset = config['roi_offset']
        roi_shape = config['roi_shape']
    else:
        roi_offset = None
        roi_shape = None

    if roi_offset is not None:
        total_roi = Roi(roi_offset, roi_shape)
    else:
        total_roi = fragments.roi

    read_roi = Roi((0,)*fragments.roi.dims, block_size)
    write_roi = Roi((0,)*fragments.roi.dims, block_size)

    lut_dir: str = os.path.join(fragments_file, lut_dir, "fragment_segment")

    logging.info(msg="Preparing segmentation dataset...")

    for threshold in thresholds:
        seg_name: str = f"{seg_dataset}/{merge_function}/{int(threshold*100)}"

        start: float = time.time()
        logging.info(fragments.roi)
        logging.info(fragments.voxel_size)
        segmentation = prepare_ds(
            filename=seg_file,
            ds_name=seg_name,
            total_roi=total_roi,
            voxel_size=voxel_size,
            dtype=np.uint64,
            write_roi=write_roi,
            compressor={"id": "blosc", "clevel": 5},
            delete=True,
        )

        lut_filename: str = f"seg_{config['edges_table']}_{int(threshold*100)}"
        lut: str = os.path.join(lut_dir, lut_filename + ".npz")

        assert os.path.exists(path=lut), f"{lut} does not exist"

        logging.info(msg="Reading fragment-segment LUT...")

        lut = np.load(file=lut)["fragment_segment_lut"]

        logging.info(msg=f"Found {len(lut[0])} fragments in LUT")

        num_segments: int = len(np.unique(ar=lut[1]))
        logging.info(msg=f"Relabelling fragments to {num_segments} segments")

        task = daisy.Task(
            task_id="ExtractSegmentsTask",
            total_roi=total_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=lambda b: segment_in_block(
                block=b, segmentation=segmentation, fragments=fragments, lut=lut
            ),
            fit="shrink",
            num_workers=num_workers,
        )

        done: bool = daisy.run_blockwise(tasks=[task])

        if not done:
            raise RuntimeError(
                "Extraction of segmentation from LUT failed for (at least) one block"
            )

        logging.info(
            msg=f"Took {time.time() - start} seconds to extract segmentation from LUT"
        )

        logging.info(
            msg=f"{seg_file} {seg_name}"
        )

    return True


def segment_in_block(block, segmentation, fragments, lut) -> None:
    logging.info(msg="Copying fragments to memory...")

    # load fragments
    fragments = fragments.to_ndarray(block.write_roi)

    # replace values, write to empty array
    relabelled: np.ndarray = np.zeros_like(fragments)
    relabelled: np.ndarray = replace_values(
        in_array=fragments, old_values=lut[0], new_values=lut[1], out_array=relabelled
    )

    segmentation[block.write_roi] = relabelled


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["processing"]["hglom_segment"] | yaml_config["db"]

    start = time.time()
    extract_segmentation(config)
    end = time.time()

    seconds = end - start
    logging.info(f'Total time to extract_segmentation: {seconds}')
