import multiprocessing
multiprocessing.set_start_method('fork')

import daisy
import json
import yaml
import logging
import subprocess
import numpy as np
import os
import sys
import time

from pathlib import Path
from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, prepare_ds

logging.basicConfig(level=logging.INFO)
scripts_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def agglomerate(
        config: dict) -> bool:
    '''

    Agglomerate in parallel blocks. Requires that affinities and supervoxels
    have been generated.

    Args:

        affs_file (``string``):

            Path to file (zarr/n5) where predictions are stored.

        affs_dataset (``string``):

            Predictions dataset to use (e.g 'volumes/affs').

        fragments_file (``string``):

            Path to file (zarr/n5) where fragments (supervoxels) are stored.

        fragments_dataset (``string``):

            Name of fragments (supervoxels) dataset (e.g 'volumes/fragments').

        rag_path (``string``):

        block_size (``tuple`` of ``int``):

            The size of one block in world units (must be multiple of voxel
            size).

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction in world units.

#        db_host (``string``):
#
#            Name of MongoDB client.
#
#        db_name (``string``):
#
#            Name of MongoDB database to use (for logging successful blocks in
#            check function and reading nodes from + writing edges to the region
#            adjacency graph).
#
        num_workers (``int``):

            How many blocks to run in parallel.

        merge_function (``string``):

            Symbolic name of a merge function. See dictionary in worker script
            (workers/agglomerate_worker.py).

    '''
    affs_file = config['affs_file']
    affs_dataset = config['affs_dataset']
    fragments_file = config['fragments_file']
    fragments_dataset = config['fragments_dataset']
    merge_function = config['merge_function']
    num_workers = config['num_workers']

    logging.info(f"Reading affs from {affs_file}")
    affs = open_ds(affs_file, affs_dataset, mode='r')

    logging.info(f"Reading fragments from {fragments_file}")
    fragments = open_ds(fragments_file, fragments_dataset, mode='r')

    # ROI
    if 'block_size' in config and config['block_size'] is not None:
        block_size = Coordinate(config["block_size"])
    else:
        block_size = fragments.chunk_shape * 4 * fragments.voxel_size

    if 'context' in config and config['context'] is not None:
        context = Coordinate(config["context"])
    else:
        context = Coordinate(fragments.chunk_shape) / 4
        context *= fragments.voxel_size

    if 'roi_offset' in config and 'roi_shape' in config:
        roi_offset = config['roi_offset']
        roi_shape = config['roi_shape']
    else:
        roi_offset = None
        roi_shape = None

    if roi_offset is not None:
        total_roi = Roi(roi_offset, roi_shape).grow(context,context)
    else:
        total_roi = fragments.roi.grow(context,context)

    read_roi = Roi((0,)*affs.roi.dims, block_size).grow(context, context)
    write_roi = Roi((0,)*affs.roi.dims, block_size)
    
    # blockwise watershed
    task = daisy.Task(
        task_id="AgglomerateTask",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda: start_worker(config),
        num_workers=num_workers,
        read_write_conflict=True,
        max_retries=20,
        fit='shrink')
    
    done: bool = daisy.run_blockwise(tasks=[task])

    if not done:
        raise RuntimeError("At least one block failed!")
    
    return done


def start_worker(config: dict):

    worker_id = daisy.Context.from_env()["worker_id"]

    logging.info(f"worker {worker_id} started...")

    config_file = Path(config["fragments_file"]) / "agglom_config.json"

    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    worker = os.path.join(scripts_dir,'workers/agglomerate_worker.py')

    subprocess.run(
        [
            "python",
            worker,
            str(config_file),
        ]
    )


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["processing"]["agglomerate"] | yaml_config["db"]

    start = time.time()
    agglomerate(config)
    end = time.time()

    seconds = end - start
    logging.info(f'Total time to agglomerate: {seconds}')
