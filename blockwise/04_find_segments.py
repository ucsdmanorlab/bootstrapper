import sys
import yaml
import os
import time
import logging
from pathlib import Path
import numpy as np

from funlib.segment.graphs.impl import connected_components
from funlib.geometry import Roi
from funlib.persistence import open_ds
from funlib.persistence.graphs import SQLiteGraphDataBase, PgSQLGraphDatabase
from funlib.persistence.types import Vec

logging.getLogger().setLevel(logging.INFO)


def find_segments(
        config: dict) -> bool:
    """
    xtract and store fragments from supervoxels and generate segmentation lookup tables.

    Args:
        affs_file (``str``):
            Path (relative or absolute) to the zarr file where affinities are stored.

        affs_dataset (``str``):
            The name of the fragments dataset to read from in the affinities file.

        fragments_file (``str``):
            Path (relative or absolute) to the zarr file where fragments are stored.

        fragments_dataset (``str``):
            The name of the fragments dataset to read from in the fragments file.

        thresholds_minmax (``list[int]``, optional):
            The lower and upper bounds to use for generating thresholds. Default is [0, 1].

        thresholds_step (``float``, optional):
            The step size to use when generating thresholds between min/max. Default is 0.02.

        merge_function (``str``, optional):
            The merge function used to create the segmentation. Default is "hist_quant_75".

    Returns:
        ``bool``:
            True if the operation was successful, False otherwise.
    """
    fragments_file = config['fragments_file']
    fragments_dataset = config['fragments_dataset']
    lut_dir = config["lut_dir"]
    thresholds_minmax = config['thresholds_minmax']
    thresholds_step = config['thresholds_step']
    merge_function = config['merge_function']

    logging.info("Reading fragments")
    start: float = time.time()

    fragments = open_ds(fragments_file, fragments_dataset)

    if 'roi_offset' in config and 'roi_shape' in config:
        roi_offset = config['roi_offset']
        roi_shape = config['roi_shape']
    else:
        roi_offset = None
        roi_shape = None

    if roi_offset is not None:
        roi = Roi(roi_offset, roi_shape)
    else:
        roi = fragments.roi

    logging.info("Opening RAG DB...")
    
    if 'db_file' in config:  
        # SQLiteGraphDatabase
        graph_provider = SQLiteGraphDataBase(
            position_attribute="center",
            db_file=Path(config['db_file']),
            mode="r",
            nodes_table=config['nodes_table'],
            edges_table=config['edges_table'],
            node_attrs={"center": Vec(int,3)},
            edge_attrs={"merge_score": float, "agglomerated": bool}
        )
        logging.info("Using SQLiteGraphDatabase")
    else:
        # PgSQLGraphDatabase
        graph_provider = PgSQLGraphDatabase(
            position_attribute="center",
            db_name=config['db_name'],
            db_host=config['db_host'],
            db_user=config['db_user'],
            db_password=config['db_password'],
            db_port=config['db_port'],
            mode="r",
            nodes_table=config['nodes_table'],
            edges_table=config['edges_table'],
            node_attrs={"center": Vec(int,3)},
            edge_attrs={"merge_score": float, "agglomerated": bool}
        )
        logging.info("Using PgSQLGraphDatabase")

    logging.info("RAG file opened")

    node_attrs: list = graph_provider.read_nodes(roi=roi)
    edge_attrs: list = graph_provider.read_edges(roi=roi, nodes=node_attrs)

    logging.info(msg=f"Read graph in {time.time() - start}")

    if "id" not in node_attrs[0]:
        logging.info(msg="No nodes found in roi %s" % roi)
        return

    nodes: list = [node["id"] for node in node_attrs]

    edge_u: list = [np.uint64(edge["u"]) for edge in edge_attrs]
    edge_v: list = [np.uint64(edge["v"]) for edge in edge_attrs]

    edges: np.ndarray = np.stack(arrays=[edge_u, edge_v], axis=1)

    scores: list = [np.float32(edge["merge_score"]) for edge in edge_attrs]

    logging.info(msg=f"Complete RAG contains {len(nodes)} nodes, {len(edges)} edges")

    out_dir: str = os.path.join(fragments_file, lut_dir, "fragment_segment")

    os.makedirs(out_dir, exist_ok=True)

    thresholds = [
        round(i, 2)
        for i in np.arange(
            float(thresholds_minmax[0]), float(thresholds_minmax[1]), thresholds_step
        )
    ]

    # parallel processing
    start = time.time()


    nodes = np.asarray(nodes,dtype=np.uint64)
    edges = np.asarray(edges,dtype=np.uint64)
    scores = np.asarray(scores)

    for t in thresholds:
        get_connected_components(
                nodes,
                edges,
                scores,
                t,
                f"{config['edges_table']}",
                out_dir,
            )

    logging.info(f"Created and stored lookup tables in {time.time() - start}")

    return True


def get_connected_components(
    nodes,
    edges,
    scores,
    threshold,
    edges_collection,
    out_dir,
):
    logging.info(f"Getting CCs for threshold {threshold}...")
    components = connected_components(nodes, edges, scores, threshold)

    logging.info(f"Creating fragment-segment LUT for threshold {threshold}...")
    lut = np.array([nodes, components])

    logging.info(f"Storing fragment-segment LUT for threshold {threshold}...")

    lookup = f"seg_{edges_collection}_{int(threshold*100)}"

    out_file = os.path.join(out_dir, lookup)

    np.savez_compressed(out_file, fragment_segment_lut=lut)


if __name__ == "__main__":

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["processing"]["hglom_segment"] | yaml_config["db"]

    start = time.time()
    find_segments(config)
    end = time.time()

    seconds = end - start
    logging.info(f'Total time to find_segments: {seconds}')
