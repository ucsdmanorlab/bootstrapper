import click
import toml
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


def find_segments(config):

    # read config
    fragments_dataset = config["fragments_dataset"]
    lut_dir = config["lut_dir"]
    thresholds_minmax = config["thresholds_minmax"]
    thresholds_step = config["thresholds_step"]
    db_config = config["db"]

    # load fragments
    logging.info("Reading fragments")
    fragments = open_ds(fragments_dataset)

    # get ROIs
    if "roi_offset" in config and "roi_shape" in config:
        roi_offset = config["roi_offset"]
        roi_shape = config["roi_shape"]
    else:
        roi_offset = None
        roi_shape = None

    if roi_offset is not None:
        roi = Roi(roi_offset, roi_shape)
    else:
        roi = fragments.roi

    # load RAG
    logging.info("Opening RAG DB...")
    if "db_file" in db_config:
        # SQLiteGraphDatabase
        graph_provider = SQLiteGraphDataBase(
            position_attribute="center",
            db_file=Path(db_config["db_file"]),
            mode="r",
            nodes_table=db_config["nodes_table"],
            edges_table=db_config["edges_table"],
            node_attrs={"center": Vec(int, 3)},
            edge_attrs={"merge_score": float, "agglomerated": bool},
        )
    else:
        # PgSQLGraphDatabase
        graph_provider = PgSQLGraphDatabase(
            position_attribute="center",
            db_name=db_config["db_name"],
            db_host=db_config["db_host"],
            db_user=db_config["db_user"],
            db_password=db_config["db_password"],
            db_port=db_config["db_port"],
            mode="r",
            nodes_table=db_config["nodes_table"],
            edges_table=db_config["edges_table"],
            node_attrs={"center": Vec(int, 3)},
            edge_attrs={"merge_score": float, "agglomerated": bool},
        )

    start = time.time()
    node_attrs: list = graph_provider.read_nodes(roi=roi)
    edge_attrs: list = graph_provider.read_edges(roi=roi, nodes=node_attrs)
    logging.info(msg=f"Read graph in {time.time() - start}")

    if "id" not in node_attrs[0]:
        logging.info(msg="No nodes found in roi %s" % roi)
        return

    # extract nodes, edges, and scores
    nodes: list = [node["id"] for node in node_attrs]

    edge_u: list = [np.uint64(edge["u"]) for edge in edge_attrs]
    edge_v: list = [np.uint64(edge["v"]) for edge in edge_attrs]
    edges: np.ndarray = np.stack(arrays=[edge_u, edge_v], axis=1)

    scores: list = [np.float32(edge["merge_score"]) for edge in edge_attrs]
    logging.info(msg=f"Complete RAG contains {len(nodes)} nodes, {len(edges)} edges")

    # create lookup tables directory
    out_dir: str = os.path.join(lut_dir, "fragment_segment")
    os.makedirs(out_dir, exist_ok=True)

    # generate thresholds
    thresholds = [
        round(i, 2)
        for i in np.arange(
            float(thresholds_minmax[0]), float(thresholds_minmax[1]), thresholds_step
        )
    ]

    # get connected components and store lookup tables
    start = time.time()

    for t in thresholds:
        get_connected_components(
            np.asarray(nodes, dtype=np.uint64),
            np.asarray(edges, dtype=np.uint64),
            np.asarray(scores),
            t,
            f"{db_config['edges_table']}",
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
    lookup = f"{edges_collection}_{str(int(threshold*100)).zfill(2)}"

    out_file = os.path.join(out_dir, lookup)
    np.savez_compressed(out_file, fragment_segment_lut=lut)


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def luts(config_file):
    """
    Find connected components of region graph and store lookup tables.
    """

    with open(config_file, "r") as f:
        config = toml.load(f)

    start = time.time()
    find_segments(config)
    end = time.time()

    seconds = end - start
    logging.info(f"Total time to find_segments: {seconds}")
