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


def find_segments(config, frags_ds_name=None):

    # read config
    fragments_dataset_prefix = config["fragments_dataset"]
    lut_dir = config["lut_dir"]
    thresholds_minmax = config.get("thresholds_minmax", [0.0, 1.0])
    thresholds_step = config.get("thresholds_step", 0.05)
    db_config = config["db"]

    if frags_ds_name is None:
        shift_name = []
        filter_fragments = config.get("filter_fragments", None)
        noise_eps = config.get("noise_eps", None)
        sigma = config.get("sigma", None)
        bias = config.get("bias", None)

        if filter_fragments is not None:
            shift_name.append(f"filt{filter_fragments}")
        if noise_eps is not None:
            shift_name.append(f"eps{noise_eps}")
        if sigma is not None:
            shift_name.append(f"sigma{'_'.join([str(x) for x in sigma])}")
        if bias is not None:
            shift_name.append(f"bias{'_'.join([str(x) for x in bias])}")
        shift_name = "--".join(shift_name)
        shift_name = f"{shift_name}--" if shift_name != "" else ""
        shift_name = f"{shift_name}minseed{config.get('min_seed_distance', 10)}"
        frags_ds_name = os.path.join(fragments_dataset_prefix, shift_name)

    # load fragments
    logging.info("Reading fragments")
    fragments = open_ds(frags_ds_name)

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

    # Load config file
    with open(config_file, "r") as f:
        toml_config = toml.load(f)

    config = toml_config | toml_config["ws_params"]
    for x in config.copy():
        if x.endswith("_params"):
            del config[x]

    start = time.time()
    find_segments(config)
    end = time.time()

    seconds = end - start
    logging.info(f"Total time to find_segments: {seconds}")

if __name__ == "__main__":
    luts()