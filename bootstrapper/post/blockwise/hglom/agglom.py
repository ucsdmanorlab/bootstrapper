import click
import time
import logging
import toml
from functools import partial
from pprint import pprint
import os

from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds
import daisy

logging.basicConfig(level=logging.INFO)


def agglomerate_in_block(affs, fragments, db_config, shift, merge_function, block):

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
    affs_data = affs.to_ndarray(block.read_roi, fill_value=0)[:3]
    fragments_data = fragments.to_ndarray(block.read_roi, fill_value=0)

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

    if shift is not None:
        sigma = shift["sigma"]
        noise_eps = shift["noise_eps"]
        bias = shift["bias"]

        shift = np.zeros_like(affs_data)

        if noise_eps is not None:
            shift += np.random.randn(*affs_data.shape) * noise_eps

        if sigma is not None:
            sigma = (0, *sigma)
            shift += gaussian_filter(affs_data, sigma=sigma) - affs_data

        if bias is not None:
            if type(bias) == float:
                bias = [bias] * affs_data.shape[0]
            else:
                assert len(bias) == affs_data.shape[0]

            shift += np.array([bias]).reshape((-1, *((1,) * (len(affs.shape) - 1))))

        affs_data += shift

    # add fake z-affinities if 2D affinities
    if affs_data.shape[0] == 2:
        affs_data = np.stack(
            [np.zeros_like(affs_data[0]), affs_data[0], affs_data[1]], axis=0
        )

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


def agglomerate(config, frags_ds_name=None):

    logging.info(f"Agglomerating fragments with config: {pprint(config)}")

    # Extract arguments from config
    affs_dataset = config["affs_dataset"]  # Name of affinities dataset
    fragments_dataset_prefix = config["fragments_dataset"]  # Name of fragments dataset
    db_config = config["db"]  # Database configuration

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
        affs_shift = {
            "sigma": sigma,
            "noise_eps": noise_eps,
            "bias": bias,
        }
        if frags_ds_name is None:
            shift_name = []
            if noise_eps is not None:
                shift_name.append(f"{noise_eps}")
            if sigma is not None:
                shift_name.append(f"{"_".join([str(x) for x in sigma[-3:]])}")
            if bias is not None:
                shift_name.append(f"{"_".join([str(x) for x in bias])}")
            shift_name = "--".join(shift_name)
            shift_name = f"{shift_name}--" if shift_name != "" else ""
            shift_name = f"{shift_name}minseed{config.get('min_seed_distance', 10)}"
            frags_ds_name = os.path.join(fragments_dataset_prefix, shift_name)
    else:
        affs_shift = None

    # Optional waterz parameters
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
    logging.info(f"Reading affs from {affs_dataset}")
    affs = open_ds(affs_dataset)
    logging.info(f"Reading fragments from {frags_ds_name}")
    fragments = open_ds(frags_ds_name)
    voxel_size = affs.voxel_size

    # get total ROI
    if roi_offset is not None:
        total_roi = Roi(roi_offset, roi_shape)
    else:
        total_roi = fragments.roi

    # get block size, context
    if blockwise:
        if block_shape is not None:
            block_size = Coordinate(block_shape) * voxel_size
        else:
            block_size = Coordinate(affs.chunk_shape[1:]) * voxel_size

        if context is not None:
            context = Coordinate(context) * voxel_size
        else:
            context = (
                Coordinate(
                    [
                        10,
                    ]
                    * affs.roi.dims
                )
                * voxel_size
            )

    else:  # blockwise is False
        block_size = total_roi.get_shape()
        context = Coordinate(
            [
                0,
            ]
            * affs.roi.dims
        )
        num_workers = 1

    # get block read ROI, write ROI
    read_roi = Roi((0,) * affs.roi.dims, block_size).grow(context, context)
    write_roi = Roi((0,) * affs.roi.dims, block_size)
    logging.info(
        f"Total ROI: {total_roi}, Read ROI: {read_roi}, Write ROI: {write_roi}"
    )

    read_roi = Roi((0,) * affs.roi.dims, block_size).grow(context, context)
    write_roi = Roi((0,) * affs.roi.dims, block_size)
    logging.info(
        f"Total ROI: {total_roi}, Read ROI: {read_roi}, Write ROI: {write_roi}"
    )

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
            affs_shift,
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

    config = toml_config | toml_config["ws_params"]
    for x in config.copy():
        if x.endswith("_params"):
            del config[x]

    start = time.time()
    agglomerate(config)
    end = time.time()

    seconds = end - start
    logging.info(f"Total time to agglomerate fragments: {seconds} ")

if __name__ == "__main__":
    agglomerate()