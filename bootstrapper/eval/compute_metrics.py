import yaml
import time
import logging
from tqdm import tqdm

import numpy as np
import networkx as nx

from funlib.segment.arrays import replace_values
from funlib.evaluate import (
    rand_voi,
    expected_run_length,
    get_skeleton_lengths,
)
from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_skeletons(gt_skeletons_file, roi):
    """
    Read skeletons from a graphml file.
    """
    skels = nx.read_graphml(gt_skeletons_file)

    bz, by, bx = roi.get_begin()
    ez, ey, ex = roi.get_end()

    # remove outside nodes and edges
    remove_nodes = []
    for node, data in skels.nodes(data=True):
        if "position_z" not in data:
            remove_nodes.append(node)
        elif "position_y" not in data:
            remove_nodes.append(node)
        elif "position_x" not in data:
            remove_nodes.append(node)
        else:
            if data["position_z"] <= bz or data["position_z"] >= ez:
                remove_nodes.append(node)
            elif data["position_y"] <= by or data["position_y"] >= ey:
                remove_nodes.append(node)
            elif data["position_x"] <= bx or data["position_x"] >= ex:
                remove_nodes.append(node)
            else:
                assert data["id"] >= 0

    logger.info(
        "Removing %s nodes out of %s in gt skeletons",
        len(remove_nodes),
        len(skels.nodes),
    )
    for node in remove_nodes:
        skels.remove_node(node)

    # remove isolated nodes
    remove_nodes.extend(list(nx.isolates(skels)))

    # return skels
    skeletons = nx.Graph()

    # Add nodes with integer identifiers and their attributes
    for node, attrs in skels.nodes(data=True):
        skeletons.add_node(int(node), **attrs)

    # Add edges with updated node identifiers
    for u, v, attrs in skels.edges(data=True):
        skeletons.add_edge(int(u), int(v), **attrs)

    return skeletons


def compute_metrics(
        seg_dataset,
        gt_labels_dataset,
        gt_skeletons_file,
        mask_dataset=None,
        roi_offset=None,
        roi_shape=None,
):
    seg_ds = open_ds(seg_dataset)
    gt_labels_ds = None if gt_labels_dataset is None else open_ds(gt_labels_dataset)
    mask_ds = None if mask_dataset is None else open_ds(mask_dataset)
    
    # get roi
    roi = seg_ds.roi
    if gt_labels_ds:
        roi = roi.intersect(gt_labels_ds.roi)
    if mask_ds:
        roi = roi.intersect(mask_ds.roi)
    if roi_offset is not None:
        roi_offset = Coordinate(roi_offset)
        roi_shape  = Coordinate(roi_shape)
        roi = Roi(roi_offset, roi_shape).intersect(roi)

    print(seg_ds.roi, roi)

    # read gt skeletons
    gt_skeletons = None if gt_skeletons_file is None else read_skeletons(gt_skeletons_file, roi) 

    # load and mask seg
    seg = seg_ds[roi]
    mask = None if mask_ds is None else mask_ds[roi]
    if mask is not None:
        seg *= mask

    metrics = {}

    # gt labels eval
    if gt_labels_ds:
        gt_labels = gt_labels_ds[roi] if mask is None else gt_labels_ds[roi] * mask
        rand_voi_report = rand_voi(gt_labels, seg, return_cluster_scores=False)
        for k in {"voi_split_i", "voi_merge_j"}:
            del rand_voi_report[k]
        metrics["voi"] = rand_voi_report

    # gt skeletons eval
    if gt_skeletons is not None:
        # get skeleton lengths
        skeleton_lengths = get_skeleton_lengths(
            gt_skeletons,
            skeleton_position_attributes=["position_z", "position_y", "position_x"],
            skeleton_id_attribute="id",
            store_edge_length="length",
        )
        total_length = np.sum([l for _, l in skeleton_lengths.items()])

        # get node segment lut
        for node in tqdm(gt_skeletons.nodes):
            try:
                gt_skeletons.nodes[node]["pred_seg_id"] = int(seg_ds[
                    Coordinate(
                        gt_skeletons.nodes[node]["position_z"],
                        gt_skeletons.nodes[node]["position_y"],
                        gt_skeletons.nodes[node]["position_x"],
                    )
                ])
            except:
                raise Exception(
                    f" node {gt_skeletons.nodes[node]} is not in seg_ds"
                )

        erl, stats = expected_run_length(
            gt_skeletons,
            skeleton_id_attribute="id",
            edge_length_attribute="length",
            node_segment_lut=nx.get_node_attributes(gt_skeletons, "pred_seg_id"),
            skeleton_lengths=skeleton_lengths,
            return_merge_split_stats=True,
        )

        max_erl, _ = expected_run_length(
            gt_skeletons,
            skeleton_id_attribute="id",
            edge_length_attribute="length",
            node_segment_lut=nx.get_node_attributes(gt_skeletons, "id"),
            skeleton_lengths=skeleton_lengths,
            return_merge_split_stats=True,
        )
        
        merge_stats = stats["merge_stats"]
        n_mergers = sum([len(v) for v in merge_stats.values()])

        merge_stats.pop(0, None)  # ignore "mergers" with background
        merge_stats.pop(0.0, None)
        n_non0_mergers = sum([len(v) for v in merge_stats.values()])

        split_stats = stats["split_stats"]
        n_splits = sum([len(v) for v in split_stats.values()])

        nerl = erl / max_erl
            
        metrics["skel"] = {
            "erl": erl,
            "nerl": nerl,
            "max_erl": max_erl,
            "total_path_length": total_length,
            "n_mergers": n_mergers,
            "n_splits": n_splits,
            "n_non0_mergers": n_non0_mergers,
        }

    return metrics