import daisy
import json
import logging
import sys
import time
import numpy as np
from pathlib import Path

import mahotas
import waterz
from funlib.segment.arrays import relabel, replace_values

from scipy.ndimage import (
    measurements,
    center_of_mass,
    maximum_filter,
    gaussian_filter,
    distance_transform_edt,
    binary_erosion
)
from skimage.measure import label
from funlib.geometry import Coordinate
from funlib.persistence import open_ds, Array
from funlib.persistence.graphs import SQLiteGraphDataBase, PgSQLGraphDatabase
from funlib.persistence.types import Vec

logging.basicConfig(level=logging.INFO)


def watershed_from_boundary_distance(
    boundary_distances, boundary_mask, return_seeds=False, id_offset=0, min_seed_distance=10
):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered == boundary_distances
    seeds, n = mahotas.label(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    fragments = mahotas.cwatershed(boundary_distances.max() - boundary_distances, seeds)

    if boundary_mask is not None:
        fragments *= boundary_mask

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret

def watershed_from_affinities(
    affs,
    max_affinity_value=1.0,
    fragments_in_xy=False,
    background_mask=False,
    mask_thresh=0.5,
    return_seeds=False,
    min_seed_distance=10,
):
    """Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.
    Returns:
        (fragments, max_id)
        or
        (fragments, max_id, seeds) if return_seeds == True"""
   
    # add random noise
    random_noise = np.random.randn(*affs.shape) * 0.01

    # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # towards processing the central regions of objects first.
    logging.info("Smoothing affs")
    smoothed_affs: np.ndarray = (
            gaussian_filter(affs, sigma=(0, 1, 2, 2))
            - 0.5
    ) * 0.05

    affs = (affs + random_noise + smoothed_affs).astype(np.float32)
    affs = np.clip(affs, 0.0, 1.0)

    if fragments_in_xy:

        mean_affs = 0.5 * (affs[-1] + affs[-2]) # affs are (c,z,y,x)
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):

            boundary_mask = mean_affs[z] > mask_thresh * max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            if background_mask is False:
                boundary_mask = None

            ret = watershed_from_boundary_distance(
                boundary_distances,
                boundary_mask,
                return_seeds=return_seeds,
                id_offset=id_offset,
                min_seed_distance=min_seed_distance,
            )

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:

        boundary_mask = np.mean(affs, axis=0) > mask_thresh * max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        if background_mask is False:
            boundary_mask = None

        ret = watershed_from_boundary_distance(
            boundary_distances, boundary_mask, return_seeds, min_seed_distance=min_seed_distance
        )

        fragments = ret[0]

    return ret

def upsample(a, factor):

    for d, f in enumerate(factor):
        a = np.repeat(a, f, axis=d)

    return a

def get_mask_data_in_roi(mask, roi, target_voxel_size):

    assert mask.voxel_size.is_multiple_of(target_voxel_size), (
        "Can not upsample from %s to %s" % (mask.voxel_size, target_voxel_size))

    aligned_roi = roi.snap_to_grid(mask.voxel_size, mode='grow')
    aligned_data = mask.to_ndarray(aligned_roi, fill_value=0)

    if mask.voxel_size == target_voxel_size:
        return aligned_data

    factor = mask.voxel_size/target_voxel_size

    upsampled_aligned_data = upsample(aligned_data, factor)

    upsampled_aligned_mask = Array(
        upsampled_aligned_data,
        roi=aligned_roi,
        voxel_size=target_voxel_size)

    return upsampled_aligned_mask.to_ndarray(roi)

def watershed_in_block(
        affs,
        block,
        context,
        rag_provider,
        fragments_out,
        num_voxels_in_block,
        mask=None,
        fragments_in_xy=False,
        background_mask=False,
        mask_thresh=0.5,
        min_seed_distance=5,
        epsilon_agglomerate=0.01,
        filter_fragments=0.0,
        replace_sections=None):
    '''Extract fragments from affinities in block using watershed.

    Args:

        affs (`class:Array`):

            An array containing affinities.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction, in world units.
        
        rag_provider (`class:RagProvider`):

            A RAG provider to write nodes for extracted fragments to. This does
            not yet add adjacency edges, for that, an agglomeration method
            should be called after this function.


        filter_fragments (float):

            Filter fragments that have an average affinity lower than this
            value.

        min_seed_distance (int):

            Controls distance between seeds in the initial watershed. Reducing
            this value improves downsampled segmentation.

        block_size (``tuple`` of ``int``):

            The size of the blocks to process in parallel in world units.
        fragments_out (`class:Array`):

            An array to store fragments in. Should be of ``dtype`` ``uint64``.

        num_workers (``int``):

            The number of parallel workers.

        mask (`class:Array`):

            A dataset containing a mask. If given, fragments are only extracted
            for masked-in (==1) areas.

        fragments_in_xy (``bool``):

            Whether to extract fragments for each xy-section separately.

        epsilon_agglomerate (``float``):

            Perform an initial waterz agglomeration on the extracted fragments
            to this threshold. Skip if 0 (default).

        filter_fragments (``float``):

            Filter fragments that have an average affinity lower than this
            value.

        replace_sections (``list`` of ``int``):

            Replace fragments data with zero in given sections (useful if large
            artifacts are causing issues). List of section numbers (in voxels)
    '''

    total_roi = affs.roi

    logging.debug("reading affs from %s", block.read_roi)

    affs = affs.intersect(block.read_roi)
    affs.materialize()

    if affs.dtype == np.uint8:
        logging.info("Assuming affinities are in [0,255]")
        affs.data = affs.data.astype(np.float32)/255.0
        max_affinity_value = 1.0
    else:
        max_affinity_value = 1.0

    if mask is not None:

        logging.debug("reading mask from %s", block.read_roi)
        mask_data = get_mask_data_in_roi(mask, affs.roi, affs.voxel_size)
        logging.debug("masking affinities")
        affs.data *= mask_data

    # extract fragments
    fragments_data, _ = watershed_from_affinities(
        affs.data,
        max_affinity_value=max_affinity_value,
        fragments_in_xy=fragments_in_xy,
        background_mask=background_mask,
        mask_thresh=mask_thresh,
        min_seed_distance=min_seed_distance,
    )

    if mask is not None:
        fragments_data *= mask_data.astype(np.uint64)

    if filter_fragments > 0:

        if fragments_in_xy:
            average_affs = np.mean(affs.data[-2:]/max_affinity_value, axis=0)
        else:
            average_affs = np.mean(affs.data/max_affinity_value, axis=0)

        filtered_fragments = []

        fragment_ids = np.unique(fragments_data)

        for fragment, mean in zip(
                fragment_ids,
                measurements.mean(
                    average_affs,
                    fragments_data,
                    fragment_ids)):
            if mean < filter_fragments:
                filtered_fragments.append(fragment)

        filtered_fragments = np.array(
            filtered_fragments,
            dtype=fragments_data.dtype)
        replace = np.zeros_like(filtered_fragments)
        replace_values(fragments_data, filtered_fragments, replace, inplace=True)

    if epsilon_agglomerate > 0:

        logging.info(
            "Performing initial fragment agglomeration until %f",
            epsilon_agglomerate)

        # add fake z-affinity channel if stacked 2D affinities
        if affs.data.shape[0] == 2:
            affs.data = np.stack([
                            np.zeros_like(affs.data[0]),
                            affs.data[-2],
                            affs.data[-1]])

        generator = waterz.agglomerate(
                affs=affs.data/max_affinity_value,
                thresholds=[epsilon_agglomerate],
                fragments=fragments_data,
                scoring_function='OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
                discretize_queue=256,
                return_merge_history=False,
                return_region_graph=False)
        fragments_data[:] = next(generator)

        # cleanup generator
        for _ in generator:
            pass

    if replace_sections:

        logging.info("Replacing sections...")

        block_begin = block.write_roi.get_begin()
        shape = block.write_roi.get_shape()

        z_context = context[0]/affs.voxel_size[0]
        logging.info("Z context: %i",z_context)

        mapping = {}

        voxel_offset = block_begin[0]/affs.voxel_size[0]

        for i,j in zip(
                range(fragments_data.shape[0]),
                range(shape[0])):
            mapping[i] = i
            mapping[j] = int(voxel_offset + i) \
                    if block_begin[0] == total_roi.get_begin()[0] \
                    else int(voxel_offset + (i - z_context))

        logging.info('Mapping: %s', mapping)

        replace = [k for k,v in mapping.items() if v in replace_sections]

        for r in replace:
            logging.info("Replacing mapped section %i with zero", r)
            fragments_data[r] = 0

    #todo add key value replacement option

    fragments = Array(fragments_data, affs.roi, affs.voxel_size)

    # crop fragments to write_roi
    fragments = fragments[block.write_roi]
    fragments.materialize()
    max_id = fragments.data.max()

    # ensure we don't have IDs larger than the number of voxels (that would
    # break uniqueness of IDs below)
    if max_id > num_voxels_in_block:
        logging.warning(
            "fragments in %s have max ID %d, relabelling...",
            block.write_roi, max_id)
        fragments.data, max_id = relabel(fragments.data)

        assert max_id < num_voxels_in_block

    # ensure unique IDs
    id_bump = block.block_id[1]*num_voxels_in_block
    logging.debug("bumping fragment IDs by %i", id_bump)
    fragments.data[fragments.data>0] += id_bump
    fragment_ids = range(id_bump + 1, id_bump + 1 + int(max_id))

    # store fragments
    logging.debug("writing fragments to %s", block.write_roi)
    fragments_out[block.write_roi] = fragments

    # following only makes a difference if fragments were found
    if max_id == 0:
        return

    # get fragment centers
    fragment_centers = {
        fragment: block.write_roi.get_offset() + affs.voxel_size*Coordinate(center)
        for fragment, center in zip(
            fragment_ids,
            measurements.center_of_mass(fragments.data, fragments.data, fragment_ids))
        if not np.isnan(center[0])
    }

    # store nodes
    rag = rag_provider[block.write_roi]
    rag.add_nodes_from([
        (node, {
            'center': c
            }
        )
        for node, c in fragment_centers.items()
    ])
    #rag_provider.write_graph(rag,block.write_roi)
    rag_provider.write_nodes(rag.nodes,block.write_roi)

def extract_fragments_worker(input_config):

    logging.info(sys.argv)

    with open(input_config, 'r') as f:
        config = json.load(f)

    logging.info(config)

    affs_file = config['affs_file']
    affs_dataset = config['affs_dataset']
    fragments_file = config['fragments_file']
    fragments_dataset = config['fragments_dataset']

    context = config['context']
    num_voxels_in_block = config['num_voxels_in_block']
    fragments_in_xy = config['fragments_in_xy']
    background_mask = config['background_mask']
    mask_thresh = config['mask_thresh']
    min_seed_distance = config['min_seed_distance']
    epsilon_agglomerate = config['epsilon_agglomerate']
    filter_fragments = config['filter_fragments']
    replace_sections = config['replace_sections']

    logging.info(f"Reading affs from {affs_file}")

    affs = open_ds(affs_file, affs_dataset, mode='r')

    logging.info(f"Reading fragments from {fragments_file}")

    fragments = open_ds(
        fragments_file,
        fragments_dataset,
        mode='r+')

    if config['mask_file']:
        logging.info(f"Reading mask from {config['mask_file']}")
        mask = open_ds(config['mask_file'], config['mask_dataset'])
    else:
        mask = None

    # open RAG DB
    logging.info("Opening RAG DB...")

    if 'rag_path' in config:  
        # SQLiteGraphDatabase
        rag_provider = SQLiteGraphDataBase(
            Path(config['rag_path']),
            position_attributes=["center_z", "center_y", "center_x"],
            mode="r+",
            nodes_table=config['nodes_table'],
            edges_table=config['edges_table'],
            edge_attrs={"merge_score": float, "agglomerated": bool}
        )
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
            mode="r+",
            nodes_table=config['nodes_table'],
            edges_table=config['edges_table'],
            node_attrs={"center": Vec(int,3)},
            edge_attrs={"merge_score": float, "agglomerated": bool}
        )
        logging.info("Using PgSQLGraphDatabase")

    logging.info("RAG DB opened")

# TODO: open block done DB
#    client = pymongo.MongoClient(db_host)
#    db = client[db_name]
#    blocks_extracted = db['blocks_extracted']

    client = daisy.Client()

    while True:

        with client.acquire_block() as block:

            if block is None:
                break

            start = time.time()

            watershed_in_block(
                affs,
                block,
                context,
                rag_provider,
                fragments,
                num_voxels_in_block=num_voxels_in_block,
                mask=mask,
                fragments_in_xy=fragments_in_xy,
                background_mask=background_mask,
                mask_thresh=mask_thresh,
                min_seed_distance=min_seed_distance,
                epsilon_agglomerate=epsilon_agglomerate,
                filter_fragments=filter_fragments,
                replace_sections=replace_sections,
            )

# TODO: block done
#            document = {
#                'num_cpus': 5,
#                'queue': queue,
#                'block_id': block.block_id,
#                'read_roi': (
#                    block.read_roi.get_begin(),
#                    block.read_roi.get_shape()
#                ),
#                'write_roi': (
#                    block.write_roi.get_begin(),
#                    block.write_roi.get_shape()
#                ),
#                'start': start,
#                'duration': time.time() - start
#            }
#
#            blocks_extracted.insert(document)


if __name__ == '__main__':

    extract_fragments_worker(sys.argv[1])
