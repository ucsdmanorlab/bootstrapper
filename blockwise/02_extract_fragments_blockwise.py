import multiprocessing
multiprocessing.set_start_method('fork')

import daisy
import subprocess
import json
import yaml
import logging
import numpy as np
import os
import pymongo
import sys
import time
import pprint

from pathlib import Path
from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, prepare_ds
from funlib.persistence.graphs import SQLiteGraphDataBase, PgSQLGraphDatabase
from funlib.persistence.types import Vec

logging.basicConfig(level=logging.INFO)

scripts_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def extract_fragments(
        config: dict) -> bool:
    '''
    Extract fragments in parallel blocks. Requires that affinities have been
    predicted before.

    When running parallel inference, the worker files are located in the setup
    directory of each experiment since that is where the training was done and
    checkpoints are located. When running watershed (and agglomeration) in
    parallel, we call a worker file which can be located anywhere. By default,
    we assume there is a workers directory inside the current directory that
    contains worker scripts (e.g `workers/extract_fragments_worker.py`).


    Args:
        affs_file (``string``):

            Path to file (zarr/n5) where predictions are stored.

        affs_dataset (``string``):

            Predictions dataset to use (e.g 'volumes/affs'). If using a scale pyramid,
            will try scale zero assuming stored in directory `s0` (e.g
            'volumes/affs/s0').

        fragments_file (``string``):

            Path to file (zarr/n5) to store fragments (supervoxels) - generally
            a good idea to store in the same place as affs.

        fragments_dataset (``string``):

            Name of dataset to write fragments (supervoxels) to (e.g
            'volumes/fragments').

        context (``tuple(int, int, int)``):

            The context to consider for fragment extraction and agglomeration, in world units.

        num_workers (``int``):
        
            How many blocks to run in parallel. Default is 10.

        fragments_in_xy (``bool``):

            Whether to extract fragments for each xy-section separately.
            Default is False (3D).

        background_mask (``bool``):

        mask_thresh (``float``):

        min_seed_distance (``int``):

        epsilon_agglomerate (``float``):

            Perform an initial waterz agglomeration on the extracted fragments
            to this threshold. Skip if 0 (default).

#        rag_path (``string``):
#
#        db_host (``string``):
#
#        db_name (``string``):
        
        mask_file (``string``, optional):

            Path to file (zarr/n5) containing mask.

        mask_dataset (``string``, optional):

            Name of mask dataset. Data should be uint8 where 1 == masked in, 0
            == masked out.

        filter_fragments (``float``, optional):

            Filter fragments that have an average affinity lower than this
            value.

        replace_sections (``list`` of ``int``, optional):

            Replace fragments data with zero in given sections (useful if large
            artifacts are causing issues). List of section numbers (in voxels).

    '''

    pprint.pp(config)

    # Extract parameters from the config dictionary
    affs_file = config['affs_file']
    affs_dataset = config['affs_dataset']
    fragments_file = config['fragments_file']
    fragments_dataset = config['fragments_dataset']
    num_workers = config['num_workers']
    fragments_in_xy = config['fragments_in_xy']
    background_mask = config['background_mask']
    mask_thresh = config['mask_thresh']
    min_seed_distance = config['min_seed_distance']
    epsilon_agglomerate = config['epsilon_agglomerate']
    filter_fragments = config['filter_fragments']
    replace_sections = config['replace_sections']

    # Optional mask
    if 'mask_file' in config and 'mask_dataset' in config:
        mask_file = config['mask_file']
        mask_dataset = config['mask_dataset']
    else:
        mask_file = None
        mask_dataset = None

    logging.info(f"Reading affs from {affs_file}")
    affs = open_ds(affs_file, affs_dataset)
    voxel_size = affs.voxel_size

    # ROI 
    if 'block_size' in config and config['block_size'] is not None:
        block_size = Coordinate(config["block_size"])
    else:
        block_size = Coordinate(affs.chunk_shape[1:]) * 2 * voxel_size

    if 'context' in config and config['context'] is not None:
        context = Coordinate(config["context"])
    else:
        context = Coordinate([0,] * affs.roi.dims)
        #context = Coordinate(affs.chunk_shape[1:]) * voxel_size / 4

    if 'roi_offset' in config and 'roi_shape' in config:
        roi_offset = config['roi_offset']
        roi_shape = config['roi_shape']
    else:
        roi_offset = None
        roi_shape = None

    if roi_offset is not None:
        total_roi = Roi(roi_offset, roi_shape)
    else:
        total_roi = affs.roi

    read_roi = Roi((0,)*affs.roi.dims, block_size).grow(context, context)
    write_roi = Roi((0,)*affs.roi.dims, block_size)

# TODO: block done db
#    client = pymongo.MongoClient(db_host)
#    db = client[db_name]
#
#    if 'blocks_extracted' not in db.list_collection_names():
#            blocks_extracted = db['blocks_extracted']
#            blocks_extracted.create_index(
#                [('block_id', pymongo.ASCENDING)],
#                name='block_id')
#    else:
#        blocks_extracted = db['blocks_extracted']

    # Prepare fragments dataset
    fragments = prepare_ds(
        filename=fragments_file,
        ds_name=fragments_dataset,
        total_roi=total_roi,
        voxel_size=voxel_size,
        dtype=np.uint64,
        write_size=write_roi.shape,
        compressor={"id": "blosc", "clevel": 5},
        delete=True,
    )

    # Get number of voxels in block
    num_voxels_in_block = (write_roi/affs.voxel_size).size

    # Create / Open RAG DB
    logging.info(msg="Creating RAG DB...")
    if 'db_file' in config:  
        # SQLiteGraphDatabase
        rag_provider = SQLiteGraphDataBase(
            position_attribute="center",
            db_file=Path(config['db_file']),
            mode="w",
            nodes_table=config['nodes_table'],
            edges_table=config['edges_table'],
            node_attrs={"center": Vec(int,affs.roi.dims)},
            edge_attrs={"merge_score": float, "agglomerated": bool}
        )
        rag_provider.con.close() 
        logging.info("Using SQLiteGraphDatabase")
    else:
        # PgSQLGraphDatabase
        rag_provider = PgSQLGraphDatabase(
            position_attribute="center",
            db_name=config['db_name'],
            db_host=config['db_host'],
            db_user=config['db_user'],
            db_password=config['db_password'],
            db_port=config['db_port'],
            mode="w",
            nodes_table=config['nodes_table'],
            edges_table=config['edges_table'],
            node_attrs={"center": Vec(int,affs.roi.dims)},
            edge_attrs={"merge_score": float, "agglomerated": bool}
        )
        rag_provider.connection.close() 
        logging.info("Using PgSQLGraphDatabase")    
    
    logging.info("RAG db created")

    # make config
    config['num_voxels_in_block'] = num_voxels_in_block

    # blockwise watershed
    task = daisy.Task(
        task_id="ExtractFragmentsTask",
        total_roi=total_roi.grow(context,context),
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda: start_worker(config),
# TODO: check block
#        check_function=lambda b: check_block(
#            blocks_extracted,
#            b),
        num_workers=num_workers,
        timeout=10,
        max_retries=20,
        fit='shrink')
    
    done: bool = daisy.run_blockwise(tasks=[task])

    if not done:
        raise RuntimeError("At least one block failed!")
    
    return done


def start_worker(config: dict):

    worker_id = daisy.Context.from_env()["worker_id"]
    logging.info(f"worker {worker_id} started...")

    config_file = Path(config["fragments_file"]) / "frags_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f)

    logging.info('Running block with config %s...'%config_file)

    worker = os.path.join(scripts_dir,'workers/extract_fragments_worker.py')
    subprocess.run(
        [
            "python",
            worker,
            str(config_file),
        ]
    )

# TODO: check block
#def check_block(blocks_extracted, block):
#
#    done = blocks_extracted.count({'block_id': block.block_id}) >= 1
#
#    return done

if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["hglom_segment"] | yaml_config["db"]

    start = time.time()
    extract_fragments(config)
    end = time.time()

    seconds = end - start
    logging.info(f'Total time to extract fragments: {seconds} ')
