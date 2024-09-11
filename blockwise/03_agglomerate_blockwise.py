import os
import sys
import time
import logging
import yaml
from functools import partial
from pprint import pprint

from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds
import daisy


def agglomerate_in_block(
        affs, 
        fragments,
        db_config,
        merge_function,
        block):
    
    # import
    from pathlib import Path
    import numpy as np
    from scipy.ndimage import gaussian_filter

    from funlib.persistence.graphs import SQLiteGraphDataBase, PgSQLGraphDatabase
    from funlib.persistence.types import Vec
    from funlib.segment.arrays import relabel

    from lsd.post.merge_tree import MergeTree

    import waterz

    # load array data
    affs_data = affs.to_ndarray(block.read_roi)
    fragments_data = fragments.to_ndarray(block.read_roi)
    
    # load RAG DB
    if "db_file" in db_config:
        # SQLiteGraphDatabase
        rag_provider = SQLiteGraphDataBase(
            db_file=Path(db_config["db_file"]),
            position_attribute="center",
            mode="r+",
            nodes_table=db_config["nodes_table"],
            edges_table=db_config["edges_table"],
            node_attrs={"center": Vec(int, affs.roi.dims)},
            edge_attrs={"merge_score": float, "agglomerated": bool},
        )
    else:
        # PgSQLGraphDatabase
        rag_provider = PgSQLGraphDatabase(
            position_attribute="center",
            db_name=db_config["db_name"],
            db_host=db_config["db_host"],
            db_user=db_config["db_user"],
            db_password=db_config["db_password"],
            db_port=db_config["db_port"],
            mode="r+",
            nodes_table=db_config["nodes_table"],
            edges_table=db_config["edges_table"],
            node_attrs={"center": Vec(int, affs.roi.dims)},
            edge_attrs={"merge_score": float, "agglomerated": bool},
        )
    
    # load RAG within block
    rag = rag_provider[block.read_roi]
    
    # waterz uses memory proportional to the max label in fragments, therefore
    # we relabel them here and use those
    fragments_relabelled, n, fragment_relabel_map = relabel(
        fragments_data, return_backwards_map=True
    )

    # convert affs to float32 ndarray with values between 0 and 1
    if affs_data.dtype == np.uint8:
        affs_data = affs_data.astype(np.float32) / 255.0
    else:
        affs_data = affs_data.astype(np.float32)

    # add random noise to affinities to break ties and prevent streaking
    random_noise = np.random.randn(*affs_data.shape) * 0.01
    
    # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # towards processing the central regions of objects first.
    smoothed_affs = (gaussian_filter(affs_data, sigma=(0, 1, 2, 2)) - 0.5) * 0.05

    affs_data = (affs_data + random_noise + smoothed_affs).astype(np.float32)
    affs_data = np.clip(affs_data, 0.005, 0.995)

    # So far, 'rag' does not contain any edges belonging to write_roi (there
    # might be a few edges from neighboring blocks, though). Run waterz until
    # threshold 0 to get the waterz RAG, which tells us which nodes are
    # neighboring. Use this to populate 'rag' with edges. Then run waterz for
    # the given threshold.

    # add fake z-affinities if 2D affinities
    if affs_data.shape[0] == 2:
        affs_data = np.stack([np.zeros_like(affs_data[0]), affs_data[0], affs_data[1]], axis=0)

    # run waterz with threshold 0 to get the initial RAG edges
    generator = waterz.agglomerate(
        affs=affs_data,
        thresholds=[0, 1.0],
        fragments=fragments_relabelled,
        scoring_function=merge_function,
        discretize_queue=256,
        return_merge_history=True,
        return_region_graph=True,
    )

    _, _, initial_rag = next(generator)
    for edge in initial_rag:
        u, v = fragment_relabel_map[edge["u"]], fragment_relabel_map[edge["v"]]
        u, v = int(u), int(v)
        rag.add_edge(u, v, merge_score=None, agglomerated=True)

    # compute edge merge scores
    _, merge_history, _ = next(generator)

    # clean up generator
    for _, _, _ in generator:
        pass

    # create merge tree from merge history
    merge_tree = MergeTree(fragment_relabel_map)
    for merge in merge_history:
        a, b, c, score = merge["a"], merge["b"], merge["c"], merge["score"]
        merge_tree.merge(
            fragment_relabel_map[a],
            fragment_relabel_map[b],
            fragment_relabel_map[c],
            score,
        )       

    # update RAG with new edge scores
    num_merged = 0
    for u, v, data in rag.edges(data=True):
        merge_score = merge_tree.find_merge(u, v)
        data["merge_score"] = merge_score
        if merge_score is not None:
            num_merged += 1

    # write edges to RAG within block write ROI
    rag_provider.write_edges(rag.nodes, rag.edges, block.write_roi)

    return 0

def agglomerate(config, db_config):

    logging.info("Agglomerating fragments with config:")

    # Extract arguments from config
    affs_file = config["affs_file"] # Path to affinities zarr container
    affs_dataset = config["affs_dataset"] # Name of affinities dataset
    fragments_file = config["fragments_file"] # Path to fragments zarr container
    fragments_dataset = config["fragments_dataset"] # Name of fragments dataset
    num_workers = config["num_workers"] # Number of workers to use

    # Optional parameters
    merge_function = config.get("merge_function", "mean")

    # get waterz merge function
    waterz_merge_function = {
        "hist_quant_10": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>",
        "hist_quant_10_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>",
        "hist_quant_25": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
        "hist_quant_25_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>",
        "hist_quant_50": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>",
        "hist_quant_50_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>",
        "hist_quant_75": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
        "hist_quant_75_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>",
        "hist_quant_90": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>",
        "hist_quant_90_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>",
        "mean": "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    }[merge_function]

    # Read affs, fragments
    logging.info(f"Reading affs from {affs_file}/{affs_dataset}")
    affs = open_ds(os.path.join(affs_file, affs_dataset))
    logging.info(f"Reading fragments from {fragments_file}/{fragments_dataset}")
    fragments = open_ds(os.path.join(fragments_file, fragments_dataset))
    voxel_size = affs.voxel_size

    # get block size, context, ROIs
    if "block_shape" in config and config["block_shape"] is not None:
        block_size = Coordinate(config["block_shape"]) * voxel_size
    else:
        block_size = Coordinate(affs.chunk_shape[1:]) * voxel_size

    if "context" in config and config["context"] is not None:
        context = Coordinate(config["context"]) * voxel_size
    else:
        context = ((block_size / voxel_size) // 4) * voxel_size

    if "roi_offset" in config and "roi_shape" in config:
        roi_offset = config["roi_offset"]
        roi_shape = config["roi_shape"]
    else:
        roi_offset = None
        roi_shape = None

    # get total ROI, read ROI, write ROI
    if roi_offset is not None:
        total_roi = Roi(roi_offset, roi_shape).grow(context, context)
    else:
        total_roi = affs.roi.grow(context, context)

    read_roi = Roi((0,) * affs.roi.dims, block_size).grow(context, context)
    write_roi = Roi((0,) * affs.roi.dims, block_size)
    logging.info(f"Total ROI: {total_roi}, Read ROI: {read_roi}, Write ROI: {write_roi}")

    print("Total ROI: ", total_roi)
    print("Read ROI: ", read_roi)
    print("Write ROI: ", write_roi)
    print("Block size: ", block_size)
    print("Context: ", context)
    print("Voxel size: ", voxel_size)
    print("process_function: ", partial(
            agglomerate_in_block,
            affs,
            fragments,
            db_config,
            waterz_merge_function,
        ))

    # prepare blockwise task
    task = daisy.Task(
        "AgglomerateFragments",
        total_roi,
        read_roi,
        write_roi,
        process_function=partial(
            agglomerate_in_block,
            affs,
            fragments,
            db_config,
            waterz_merge_function,
        ),
        read_write_conflict=True,
        num_workers=num_workers,
        max_retries=5,
        fit="shrink",
    )

    # Run blockwise
    ret = daisy.run_blockwise([task])

    if ret:
        print("Ran all blocks successfully!")
    else:
        print("Did not run all blocks successfully...")

    
if __name__ == "__main__":
    config_file = sys.argv[1] # Path to config file

    # Load config file
    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["hglom_segment"]
    db_config = yaml_config["db"]

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    start = time.time()
    agglomerate(config, db_config)
    end = time.time()

    seconds = end - start
    logging.info(f"Total time to agglomerate fragments: {seconds} ")