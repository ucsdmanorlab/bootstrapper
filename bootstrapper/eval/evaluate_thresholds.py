from funlib.segment.arrays import replace_values
from funlib.evaluate import (
    rand_voi,
    expected_run_length,
    get_skeleton_lengths,
    split_graph,
)
from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds
from funlib.persistence.graphs import PgSQLGraphDatabase
from funlib.persistence.types import Vec

import yaml
import logging
import multiprocessing as mp
import networkx as nx
import numpy as np
import os
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.setrecursionlimit(10000)


class EvaluateAnnotations:

    def __init__(
        self,
        gt_labels_file,
        gt_labels_ds,
        gt_skeleton_path,
        fragments_file,
        fragments_dataset,
        rag_db_config,
        lut_dir,
        roi_offset,
        roi_shape,
        compute_mincut_metric,
        thresholds_minmax=[0.05, 0.8],
        thresholds_step=0.05,
        **kwargs,
    ):

        self.labels = open_ds(os.path.join(gt_labels_file, gt_labels_ds))
        self.skeletons_file = gt_skeleton_path
        self.fragments = open_ds(os.path.join(fragments_file, fragments_dataset))
        self.db_config = rag_db_config
        self.lut_dir = os.path.join(fragments_file, lut_dir, "fragment_segment")
        if roi_offset is not None:
            self.roi = Roi(roi_offset, roi_shape)
        else:
            self.roi = self.labels.roi.intersect(self.fragments.roi)
        self.voxel_size = self.labels.voxel_size
        self.compute_mincut_metric = compute_mincut_metric
        self.thresholds_minmax = thresholds_minmax
        self.thresholds_step = thresholds_step

    def prepare_for_roi(self):

        logger.info("Preparing evaluation for ROI %s...", self.roi)

        self.skeletons = self.read_skeletons()

        # array with site IDs
        self.site_ids = np.array([n for n in self.skeletons.nodes()], dtype=np.uint64)

        # array with component ID for each site
        self.site_component_ids = np.array(
            [data["id"] for _, data in self.skeletons.nodes(data=True)]
        )
        assert self.site_component_ids.min() >= 0
        self.site_component_ids = self.site_component_ids.astype(np.uint64)
        self.number_of_components = np.unique(self.site_component_ids).size

        logger.info("Calculating skeleton lengths...")
        start = time.time()
        self.skeleton_lengths = get_skeleton_lengths(
            self.skeletons,
            skeleton_position_attributes=["position_z", "position_y", "position_x"],
            skeleton_id_attribute="id",
            store_edge_length="length",
        )

        logger.info("%.3fs", time.time() - start)

        self.total_length = np.sum([l for _, l in self.skeleton_lengths.items()])

    def prepare_for_fragments(self):
        """Get the fragment ID for each site in site_ids."""

        site_fragment_lut, num_bg_sites = get_site_fragment_lut(
            self.fragments, self.skeletons.nodes(data=True)
        )
        self.num_bg_sites = num_bg_sites

        assert site_fragment_lut.dtype == np.uint64
        logger.info("Found %d sites in site-fragment LUT", len(site_fragment_lut[0]))

        # convert to dictionary
        site_fragment_lut = {
            site: fragment
            for site, fragment in zip(site_fragment_lut[0], site_fragment_lut[1])
        }

        # create fragment ID array congruent to site_ids
        self.site_fragment_ids = np.array(
            [
                site_fragment_lut[s] if s in site_fragment_lut else 0
                for s in self.site_ids
            ],
            dtype=np.uint64,
        )

    def read_skeletons(self):

        skels = nx.read_graphml(self.skeletons_file)

        bz, by, bx = self.roi.get_begin()
        ez, ey, ex = self.roi.get_end()

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
                if data["position_z"] < bz or data["position_z"] > ez:
                    remove_nodes.append(node)
                elif data["position_y"] < by or data["position_y"] > ey:
                    remove_nodes.append(node)
                elif data["position_x"] < bx or data["position_x"] > ex:
                    remove_nodes.append(node)
                else:
                    assert data["id"] >= 0

        # remove isolated nodes
        remove_nodes.extend(list(nx.isolates(skels)))

        logger.info(
            "Removing %s nodes out of %s in gt skeletons",
            len(remove_nodes),
            len(skels.nodes),
        )
        for node in remove_nodes:
            skels.remove_node(node)

        # return skels
        skeletons = nx.Graph()

        # Add nodes with integer identifiers and their attributes
        for node, attrs in skels.nodes(data=True):
            skeletons.add_node(int(node), **attrs)

        # Add edges with updated node identifiers
        for u, v, attrs in skels.edges(data=True):
            skeletons.add_edge(int(u), int(v), **attrs)

        return skeletons

    def evaluate(self):

        self.prepare_for_roi()

        self.prepare_for_fragments()

        thresholds = [
            float(round(i, 2))
            for i in np.arange(
                float(self.thresholds_minmax[0]),
                float(self.thresholds_minmax[1]),
                self.thresholds_step,
            )
        ]

        results = {}
        logger.info("Evaluating thresholds...")

        with mp.get_context("spawn").Pool(len(thresholds), maxtasksperchild=1) as pool:
            results = {}
            result_objects = [
                pool.apply_async(self.evaluate_threshold, args=(threshold,))
                for threshold in thresholds
            ]

            for threshold, result_obj in zip(thresholds, result_objects):
                results[threshold] = result_obj.get()

        results = convert_dtypes(results)

        # get best result
        best_nvi_thresh = sorted(
            [(results[thresh]["nvi_sum"], thresh) for thresh in results.keys()]
        )
        best_edits_thresh = sorted(
            [
                (
                    results[thresh]["total_splits_needed_to_fix_merges"]
                    + results[thresh]["total_merges_needed_to_fix_splits"],
                    thresh,
                )
                for thresh in results.keys()
            ]
        )

        best_nvi_thresh = best_nvi_thresh[0][1]
        best_edits_thresh = best_edits_thresh[0][1]

        results = {"best_nvi": results[best_nvi_thresh]} | {
            "best_edits": results[best_edits_thresh]
        }
        return results

    def get_site_segment_ids(self, threshold):

        # get fragment-segment LUT
        lut_name = f"thresh_{int(threshold*100)}.npy"
        lut_name = f"seg_{self.db_config['edges_table']}_{int(threshold*100)}.npz"
        fragment_segment_lut = np.load(os.path.join(self.lut_dir, lut_name))[
            "fragment_segment_lut"
        ]
        assert fragment_segment_lut.dtype == np.uint64

        # get the segment ID for each site
        logger.info("Mapping sites to segments...")
        start = time.time()
        site_mask = np.isin(fragment_segment_lut[0], self.site_fragment_ids)
        site_segment_ids = replace_values(
            self.site_fragment_ids,
            fragment_segment_lut[0][site_mask],
            fragment_segment_lut[1][site_mask],
        )
        logger.info("%.3fs", time.time() - start)

        return site_segment_ids, fragment_segment_lut

    def compute_expected_run_length(self, site_segment_ids):

        logger.info("Calculating expected run length...")
        start = time.time()

        node_segment_lut = {
            site: segment for site, segment in zip(self.site_ids, site_segment_ids)
        }

        erl, stats = expected_run_length(
            skeletons=self.skeletons,
            skeleton_id_attribute="id",
            edge_length_attribute="length",
            node_segment_lut=node_segment_lut,
            skeleton_lengths=self.skeleton_lengths,
            return_merge_split_stats=True,
        )

        perfect_lut = {
            node: data["id"] for node, data in self.skeletons.nodes(data=True)
        }

        max_erl, _ = expected_run_length(
            skeletons=self.skeletons,
            skeleton_id_attribute="id",
            edge_length_attribute="length",
            node_segment_lut=perfect_lut,
            skeleton_lengths=self.skeleton_lengths,
            return_merge_split_stats=True,
        )

        split_stats = [
            {"comp_id": int(comp_id), "seg_ids": [(int(a), int(b)) for a, b in seg_ids]}
            for comp_id, seg_ids in stats["split_stats"].items()
        ]
        merge_stats = [
            {"seg_id": int(seg_id), "comp_ids": [int(comp_id) for comp_id in comp_ids]}
            for seg_id, comp_ids in stats["merge_stats"].items()
        ]

        logger.info("%.3fs", time.time() - start)

        return erl, max_erl, split_stats, merge_stats

    def compute_splits_merges_needed(
        self,
        fragment_segment_lut,
        site_segment_ids,
        split_stats,
        merge_stats,
        threshold,
    ):

        total_splits_needed = 0
        total_additional_merges_needed = 0
        total_unsplittable_fragments = []

        logger.info("Computing min-cut metric for each merging segment...")

        for i, merge in enumerate(merge_stats):
            (splits_needed, additional_merges_needed, unsplittable_fragments) = (
                self.mincut_metric(
                    fragment_segment_lut,
                    site_segment_ids,
                    merge["seg_id"],
                    merge["comp_ids"],
                    threshold,
                )
            )
            total_splits_needed += splits_needed
            total_additional_merges_needed += additional_merges_needed
            total_unsplittable_fragments += unsplittable_fragments

        total_merges_needed = 0
        for split in split_stats:
            total_merges_needed += len(split["seg_ids"]) - 1
        total_merges_needed += total_additional_merges_needed

        return (total_splits_needed, total_merges_needed, total_unsplittable_fragments)

    def mincut_metric(
        self,
        fragment_segment_lut,
        site_segment_ids,
        segment_id,
        component_ids,
        threshold,
    ):

        # get RAG for segment ID
        rag = self.get_segment_rag(segment_id, fragment_segment_lut, threshold)

        logger.info("Preparing RAG for split_graph call")
        start = time.time()

        # replace merge_score with weight
        for _, _, data in rag.edges(data=True):
            # print(_, _, data)
            data["weight"] = 1.0 - data["merge_score"]

        # find fragments for each component in segment_id
        component_fragments = {}

        # True for every site that maps to segment_id
        segment_mask = site_segment_ids == segment_id

        # print('Component ids: ', component_ids)
        # print('Self site component ids: ', self.site_component_ids)

        for component_id in component_ids:

            # print('Component id: ', component_id)

            # print('Site fragment ids: ', self.site_fragment_ids)

            # limit following to sites that are part of component_id and
            # segment_id
            component_mask = self.site_component_ids == component_id
            fg_mask = self.site_fragment_ids != 0
            mask = np.logical_and(np.logical_and(component_mask, segment_mask), fg_mask)
            site_ids = self.site_ids[mask]
            site_fragment_ids = self.site_fragment_ids[mask]

            component_fragments[component_id] = site_fragment_ids

            # print('Site ids: ', site_ids)
            # print('Site fragment ids: ', site_fragment_ids)

            for site_id, fragment_id in zip(site_ids, site_fragment_ids):

                if fragment_id == 0:
                    continue

                # For each fragment containing a site, we need a position for
                # the split_graph call. We just take the position of the
                # skeleton node that maps to it, if there are several, we take
                # the last one.

                # print('Site id: ', site_id)
                # print('Fragment id: ', fragment_id, type(fragment_id))

                site_data = self.skeletons.nodes[site_id]
                fragment = rag.nodes[fragment_id]
                fragment["position_z"] = site_data["position_z"]
                fragment["position_y"] = site_data["position_y"]
                fragment["position_x"] = site_data["position_x"]

                # Keep track of how many components share a fragment. If it is
                # more than one, this fragment is unsplittable.
                if "component_ids" not in fragment:
                    fragment["component_ids"] = set()
                fragment["component_ids"].add(component_id)

        # find all unsplittable fragments...
        unsplittable_fragments = []
        for fragment_id, data in rag.nodes(data=True):
            if fragment_id == 0:
                continue
            if "component_ids" in data and len(data["component_ids"]) > 1:
                unsplittable_fragments.append(fragment_id)
        # ...and remove them from the component lists
        for component_id in component_ids:

            fragment_ids = component_fragments[component_id]
            valid_mask = np.logical_not(np.isin(fragment_ids, unsplittable_fragments))
            valid_fragment_ids = fragment_ids[valid_mask]
            if len(valid_fragment_ids) > 0:
                component_fragments[component_id] = valid_fragment_ids
            else:
                del component_fragments[component_id]

        logger.info(
            "%d fragments are merging and can not be split", len(unsplittable_fragments)
        )

        if len(component_fragments) <= 1:
            logger.info(
                "after removing unsplittable fragments, there is nothing to "
                "do anymore"
            )
            return 0, 0, unsplittable_fragments

        # these are the fragments that need to be split
        split_fragments = list(component_fragments.values())

        logger.info("Preparation took %.3fs", time.time() - start)

        logger.info(
            "Splitting segment into %d components with sizes %s",
            len(split_fragments),
            [len(c) for c in split_fragments],
        )

        logger.info("Calling split_graph...")
        start = time.time()

        # call split_graph
        num_splits_needed = split_graph(
            rag,
            split_fragments,
            position_attributes=["position_z", "position_y", "position_x"],
            weight_attribute="weight",
            split_attribute="split",
        )

        logger.info("split_graph took %.3fs", time.time() - start)

        logger.info("%d splits needed for segment %d", num_splits_needed, segment_id)

        # get number of additional merges needed after splitting the current
        # segment
        #
        # this is the number of split labels per component minus 1
        additional_merges_needed = 0
        for component, fragments in component_fragments.items():
            split_ids = np.unique([rag.nodes[f]["split"] for f in fragments])
            additional_merges_needed += len(split_ids) - 1

        logger.info(
            "%d additional merges needed to join components again",
            additional_merges_needed,
        )

        return (num_splits_needed, additional_merges_needed, unsplittable_fragments)

    def get_segment_rag(self, segment_id, fragment_segment_lut, threshold):

        logger.info("Reading RAG for segment %d", segment_id)
        start = time.time()

        rag_provider = PgSQLGraphDatabase(
            **self.db_config,
            position_attribute="center",
            mode="r",
            node_attrs={"center": Vec(int, 3)},
            edge_attrs={"merge_score": float, "agglomerated": bool},
        )

        # get all fragments for the given segment
        segment_mask = fragment_segment_lut[1] == segment_id
        fragment_ids = fragment_segment_lut[0][segment_mask]

        # get the RAG containing all fragments
        nodes = [
            {"id": fragment_id, "segment_id": segment_id}
            for fragment_id in fragment_ids
        ]
        edges = rag_provider.read_edges(self.roi, nodes=nodes)

        logger.info("RAG contains %d nodes/%d edges", len(nodes), len(edges))

        rag = nx.Graph()
        node_list = [(n["id"], {"segment_id": n["segment_id"]}) for n in nodes]
        #        edge_list = [
        #            (e['u'], e['v'], {'merge_score': e['merge_score']})
        #            for e in edges
        #            if e['merge_score'] <= threshold
        #        ]
        edge_list = []
        for e in edges:
            if e["merge_score"] is None:
                edge_list.append((int(e["u"]), int(e["v"]), {"merge_score": 1.0}))
            else:
                if e["merge_score"] <= threshold:
                    edge_list.append(
                        (int(e["u"]), int(e["v"]), {"merge_score": e["merge_score"]})
                    )

        rag.add_nodes_from(node_list)
        rag.add_edges_from(edge_list)
        rag.remove_nodes_from(
            [n for n, data in rag.nodes(data=True) if "segment_id" not in data]
        )

        logger.info(
            "after filtering dangling node and not merged edges "
            "RAG contains %d nodes/%d edges",
            rag.number_of_nodes(),
            rag.number_of_edges(),
        )

        logger.info("Reading RAG took %.3fs", time.time() - start)

        return rag

    #    def compute_rand_voi(
    #            self,
    #            site_component_ids,
    #            site_segment_ids,
    #            return_cluster_scores=False):
    #
    #        logger.info("Computing RAND and VOI...")
    #        start = time.time()
    #
    #        rand_voi_report = rand_voi(
    #            np.array([[site_component_ids]]),
    #            np.array([[site_segment_ids]]),
    #            return_cluster_scores=return_cluster_scores)
    #
    #        logger.info("VOI split: %f", rand_voi_report['voi_split'])
    #        logger.info("VOI merge: %f", rand_voi_report['voi_merge'])
    #        logger.info("%.3fs", time.time() - start)
    #
    #        return rand_voi_report

    def compute_rand_voi(self, threshold, return_cluster_scores=False):
        # return_cluster_scores=True):

        lut_name = f"seg_{self.db_config['edges_table']}_{int(threshold*100)}.npz"
        fragment_segment_lut = np.load(os.path.join(self.lut_dir, lut_name))[
            "fragment_segment_lut"
        ]

        site_mask = np.isin(fragment_segment_lut[0], self.site_fragment_ids)
        seg = replace_values(
            self.fragments.to_ndarray(self.roi),
            fragment_segment_lut[0][site_mask],
            fragment_segment_lut[1][site_mask],
        )

        labels = self.labels.to_ndarray(self.roi)

        # ensure same shape
        if seg.shape != labels.shape:
            l_z, l_y, l_x = labels.shape[-3:]
            s_z, s_y, s_x = seg.shape[-3:]
            c_z, c_y, c_x = (min(l_z, s_z), min(l_y, s_y), min(l_x, s_x))

            labels = labels[:c_z, :c_y, :c_x]
            seg = seg[:c_z, :c_y, :c_x]

        # eval
        metrics = rand_voi(
            labels, seg * (labels > 0), return_cluster_scores=return_cluster_scores
        )

        return metrics

    def evaluate_threshold(self, threshold):

        site_segment_ids, fragment_segment_lut = self.get_site_segment_ids(threshold)

        number_of_segments = np.unique(site_segment_ids).size

        erl, max_erl, split_stats, merge_stats = self.compute_expected_run_length(
            site_segment_ids
        )

        number_of_split_skeletons = len(split_stats)
        number_of_merging_segments = len(merge_stats)

        # print('ERL: ', erl)
        # print('Max ERL: ', max_erl)
        # print('Total path length: ', self.total_length)

        normalized_erl = erl / max_erl
        # print('Normalized ERL: ', normalized_erl)

        if self.compute_mincut_metric:

            splits_needed, merges_needed, unsplittable_fragments = (
                self.compute_splits_merges_needed(
                    fragment_segment_lut,
                    site_segment_ids,
                    split_stats,
                    merge_stats,
                    threshold,
                )
            )

            average_splits_needed = splits_needed / number_of_segments
            average_merges_needed = merges_needed / self.number_of_components
        #            print(
        #                    'Number of splits needed: ', splits_needed, '\n',
        #                    'Number of merges needed: ', merges_needed, '\n',
        #                    'Number of background sites: ', self.num_bg_sites, '\n',
        #                    'Average splits needed: ', average_splits_needed, '\n',
        #                    'Average merges needed: ', average_merges_needed, '\n',
        #                    'Number of unsplittable fragments: ', len(unsplittable_fragments)
        #                )

        rand_voi_report = self.compute_rand_voi(threshold)
        #            self.site_component_ids,
        #            site_segment_ids,
        #            return_cluster_scores=True)

        report = rand_voi_report.copy()
        # report['merge_stats'] = merge_stats
        # report['split_stats'] = split_stats

        for k in {"voi_split_i", "voi_merge_j"}:
            del report[k]

        report["voi_sum"] = report["voi_split"] + report["voi_merge"]
        report["nvi_sum"] = report["nvi_split"] + report["nvi_merge"]

        report["expected_run_length"] = erl
        report["max_erl"] = max_erl
        report["total path length"] = self.total_length
        report["normalized_erl"] = normalized_erl
        report["number_of_segments"] = number_of_segments
        report["number_of_components"] = self.number_of_components
        report["number_of_merging_segments"] = number_of_merging_segments
        report["number_of_split_skeletons"] = number_of_split_skeletons

        if self.compute_mincut_metric:
            report["total_splits_needed_to_fix_merges"] = splits_needed
            report["average_splits_needed_to_fix_merges"] = average_splits_needed
            report["total_merges_needed_to_fix_splits"] = merges_needed
            report["average_merges_needed_to_fix_splits"] = average_merges_needed
            report["number_of_unsplittable_fragments"] = len(unsplittable_fragments)
            report["number_of_background_sites"] = self.num_bg_sites

        # print(threshold, report)
        report["threshold"] = threshold

        # find_worst_split_merges(report)
        return report


def get_site_fragment_lut(fragments, sites):
    """Get the fragment IDs of all the sites that are contained in the given
    ROI."""

    sites = list(sites)

    if len(sites) == 0:
        logger.info("No sites in %s, skipping", roi)
        return None, None

    start = time.time()
    fragments.materialize()
    fragment_ids = np.array(
        [
            fragments[
                Coordinate(site["position_z"], site["position_y"], site["position_x"])
            ]
            for _, site in sites
        ]
    )
    site_ids = np.array([site for site, _ in sites], dtype=np.uint64)

    fg_mask = fragment_ids != 0
    fragment_ids = fragment_ids[fg_mask]
    site_ids = site_ids[fg_mask]

    logger.info(
        "Got fragment IDs for %d sites in %.3fs", len(fragment_ids), time.time() - start
    )

    lut = np.array([site_ids, fragment_ids])

    return lut, (fg_mask == 0).sum()


def find_worst_split_merges(rand_voi_report):

    # get most severe splits/merges
    splits = sorted([(s, i) for (i, s) in rand_voi_report["voi_split_i"].items()])
    merges = sorted([(s, j) for (j, s) in rand_voi_report["voi_merge_j"].items()])

    logger.info("10 worst splits:")
    for s, i in splits[-10:]:
        logger.info("\tcomponent %d\tVOI split %.5f" % (i, s))

    logger.info("10 worst merges:")
    for s, i in merges[-10:]:
        logger.info("\tsegment %d\tVOI merge %.5f" % (i, s))


def convert_dtypes(d):
    if isinstance(d, dict):
        return {k: convert_dtypes(v) for k, v in d.items()}
    elif isinstance(d, (np.int64, np.uint64, np.int32, np.uint32)):
        return int(d)
    elif isinstance(d, (np.float64, np.float32, np.float16)):
        return float(d)
    return d


if __name__ == "__main__":

    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["evaluation"] | {"rag_db_config": yaml_config["db"]}

    evaluate = EvaluateAnnotations(**config)
    result = evaluate.evaluate()

    print(result)
