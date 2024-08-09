import daisy
import json
import logging
import sys
import time

import numpy as np
from lsd.post.merge_tree import MergeTree
from funlib.segment.arrays import relabel
from scipy.ndimage import gaussian_filter
import waterz

from pathlib import Path
from funlib.geometry import Coordinate
from funlib.persistence import open_ds
from funlib.persistence.graphs import SQLiteGraphDataBase, PgSQLGraphDatabase
from funlib.persistence.types import Vec

logging.basicConfig(level=logging.INFO)


def agglomerate_in_block(
        affs,
        fragments,
        rag_provider,
        block,
        merge_function,
        threshold):

    logging.info(
        "Agglomerating in block %s with context of %s",
        block.write_roi, block.read_roi)

    # get the sub-{affs, fragments, graph} to work on
    affs = affs.intersect(block.read_roi)
    fragments = fragments.to_ndarray(affs.roi, fill_value=0)
    rag = rag_provider[affs.roi]

    # waterz uses memory proportional to the max label in fragments, therefore
    # we relabel them here and use those
    fragments_relabelled, n, fragment_relabel_map = relabel(
        fragments,
        return_backwards_map=True)

    logging.debug("affs shape: %s", affs.shape)
    logging.debug("fragments shape: %s", fragments.shape)
    logging.debug("fragments num: %d", n)

    # convert affs to float32 ndarray with values between 0 and 1
    affs = affs.to_ndarray()[0:3]
    if affs.dtype == np.uint8:
        affs = affs.astype(np.float32)/255.0

    # add random noise
    random_noise = np.random.randn(*affs.shape) * 0.01

    # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # towards processing the central regions of objects first.
    logging.info("Smoothing affs")
    smoothed_affs = (
            gaussian_filter(affs, sigma=(0, 1, 2, 2))
            - 0.5
    ) * 0.05

    affs = (affs + random_noise + smoothed_affs).astype(np.float32)
    affs = np.clip(affs, 0.005, 0.995)
    logging.info("Smoothing affs done")

    # So far, 'rag' does not contain any edges belonging to write_roi (there
    # might be a few edges from neighboring blocks, though). Run waterz until
    # threshold 0 to get the waterz RAG, which tells us which nodes are
    # neighboring. Use this to populate 'rag' with edges. Then run waterz for
    # the given threshold.
   
    # add fake z-affinities
    if affs.shape[0] == 2:
        affs = np.stack([
                        0.5*np.ones_like(affs[0]),
                        affs[-2],
                        affs[-1]])

    # for efficiency, we create one waterz call with both thresholds
    generator = waterz.agglomerate(
            affs=affs,
            thresholds=[0, threshold],
            fragments=fragments_relabelled,
            scoring_function=merge_function,
            discretize_queue=256,
            return_merge_history=True,
            return_region_graph=True)

    # add edges to RAG
    _, _, initial_rag = next(generator)
    for edge in initial_rag:
        u, v = fragment_relabel_map[edge['u']], fragment_relabel_map[edge['v']]
        u, v = int(u), int(v)
        # this might overwrite already existing edges from neighboring blocks,
        # but that's fine, we only write attributes for edges within write_roi
        rag.add_edge(u, v, merge_score=None, agglomerated=True)

    # agglomerate fragments using affs
    _, merge_history, _ = next(generator)

    # cleanup generator
    for _, _, _ in generator:
        pass

    # create a merge tree from the merge history
    merge_tree = MergeTree(fragment_relabel_map)
    for merge in merge_history:

        a, b, c, score = merge['a'], merge['b'], merge['c'], merge['score']
        merge_tree.merge(
            fragment_relabel_map[a],
            fragment_relabel_map[b],
            fragment_relabel_map[c],
            score)

    # mark edges in original RAG with score at time of merging
    logging.debug("marking merged edges...")
    num_merged = 0
    for u, v, data in rag.edges(data=True):
        merge_score = merge_tree.find_merge(u, v)
        data['merge_score'] = merge_score
        if merge_score is not None:
            num_merged += 1

    logging.info("merged %d edges", num_merged)

    # write back results (only within write_roi)
    logging.debug("writing to DB...")
    rag_provider.write_edges(rag.nodes,rag.edges,block.write_roi)


def agglomerate_worker(input_config):

    logging.info(sys.argv)

    with open(input_config, 'r') as f:
        config = json.load(f)

    logging.info(config)

    affs_file = config['affs_file']
    affs_dataset = config['affs_dataset']
    fragments_file = config['fragments_file']
    fragments_dataset = config['fragments_dataset']
    merge_function = config['merge_function']

    waterz_merge_function = {
        'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
        'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
        'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
        'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
        'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
        'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
        'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
        'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
        'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
        'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
        'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
    }[merge_function]

    logging.info(f"Reading affs from {affs_file}")
    affs = open_ds(affs_file, affs_dataset)

    logging.info(f"Reading fragments from {fragments_file}")
    fragments = open_ds(fragments_file, fragments_dataset)

    # open RAG DB
    logging.info("Opening RAG DB...")
    
    if 'db_file' in config:  
        # SQLiteGraphDatabase
        rag_provider = SQLiteGraphDataBase(
            db_file=Path(config['db_file']),
            position_attribute="center",
            mode="r+",
            nodes_table=config['nodes_table'],
            edges_table=config['edges_table'],
            node_attrs={"center": Vec(int,affs.roi.dims)},
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
            node_attrs={"center": Vec(int,affs.roi.dims)},
            edge_attrs={"merge_score": float, "agglomerated": bool}
        )
        logging.info("Using PgSQLGraphDatabase")

    logging.info("RAG DB opened")

# TODO: block done
#    # open block done DB
#    client = pymongo.MongoClient(db_host)
#    db = client[db_name]
#    blocks_agglomerated = db['blocks_agglomerated_' + merge_function]

    client = daisy.Client()

    while True:

        with client.acquire_block() as block:

            if block is None:
                break

            start = time.time()

            agglomerate_in_block(
                    affs,
                    fragments,
                    rag_provider,
                    block,
                    merge_function=waterz_merge_function,
                    threshold=1.0)

# TODO: block done
#            document = {
#                'num_cpus': 5,
#                'queue': queue,
#                'block_id': block.block_id,
#                'read_roi': (block.read_roi.get_begin(), block.read_roi.get_shape()),
#                'write_roi': (block.write_roi.get_begin(), block.write_roi.get_shape()),
#                'start': start,
#                'duration': time.time() - start
#            }
#
#            blocks_agglomerated.insert(document)


if __name__ == '__main__':

    agglomerate_worker(sys.argv[1])
