import logging
from contextlib import contextmanager
from typing import Annotated, Callable, Generator, Literal

import numpy as np
from daisy import Block
from funlib.geometry import Coordinate, Roi
from funlib.persistence.graphs.graph_database import GraphDataBase
from pydantic import Field

from volara.blockwise import BlockwiseTask
from volara.datasets import Dataset, Labels, Raw
from volara.dbs import PostgreSQL, SQLite
from volara.utils import PydanticCoordinate

from ..merge_tree import MergeTree

logger = logging.getLogger(__name__)

# waterz scoring function strings, keyed by friendly name. Only "mean" is
# enabled: it is the one the ZettaAI/waterz build precompiles (bin256). The
# quantile scorers below fall back to waterz's witty JIT, which is broken on
# this stack; uncomment them once that build path works.
WATERZ_MERGE_FUNCTIONS = {
    "mean": "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    # "hist_quant_10": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>",
    # "hist_quant_10_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>",
    # "hist_quant_25": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
    # "hist_quant_25_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>",
    # "hist_quant_50": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>",
    # "hist_quant_50_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>",
    # "hist_quant_75": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
    # "hist_quant_75_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>",
    # "hist_quant_90": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>",
    # "hist_quant_90_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>",
}


class WaterzAgglom(BlockwiseTask):
    """
    A blockwise task that scores supervoxel RAG edges with waterz.

    For each block it runs waterz agglomeration on the fragments, builds a merge
    tree from the merge history, and writes each RAG edge's merge score into the
    volara graph database (SQLite or PostgreSQL). The global segmentation is then
    obtained by thresholded connected components on the stored graph.
    """

    task_type: Literal["waterz-agglom"] = "waterz-agglom"
    db: Annotated[
        PostgreSQL | SQLite,
        Field(discriminator="db_type"),
    ]
    """
    The database containing the fragment supervoxel nodes; merge-score edges are
    added to it.
    """
    affs_data: Raw
    """
    The affinities used to score edges (only the first three nearest-neighbor
    channels are used by waterz).
    """
    frags_data: Labels
    """
    The labels array containing the supervoxels to agglomerate.
    """
    block_size: PydanticCoordinate
    context: PydanticCoordinate
    merge_function: str = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>"
    """
    The waterz scoring function string (see WATERZ_MERGE_FUNCTIONS).
    """

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False

    @property
    def task_name(self) -> str:
        return f"{self.db.id}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.frags_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(self.roi)
        return total_roi

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.frags_data.array("r").voxel_size

    @property
    def context_size(self) -> Coordinate:
        return self.context * self.frags_data.array("r").voxel_size

    @property
    def output_datasets(self) -> list[Dataset]:
        return []

    def drop_artifacts(self):
        self.db.drop_edges()

    def init(self) -> None:
        self.db.init()

    def agglomerate_in_block(self, block, affs, frags, rag_provider):
        import waterz
        from funlib.segment.arrays import relabel

        affs_data = affs.to_ndarray(block.read_roi, fill_value=0)[:3]
        frags_data = frags.to_ndarray(block.read_roi, fill_value=0)
        rag = rag_provider[block.read_roi]

        # waterz memory scales with the max fragment id, so relabel to a dense
        # range and map results back through the returned backwards map.
        frags_relabelled, _, relabel_map = relabel(
            frags_data, return_backwards_map=True
        )

        if affs_data.dtype == np.uint8:
            affs_data = affs_data.astype(np.float32) / 255.0
        else:
            affs_data = affs_data.astype(np.float32)

        # waterz expects 3 (z, y, x) affinity channels
        if affs_data.shape[0] == 2:
            affs_data = np.stack(
                [np.zeros_like(affs_data[0]), affs_data[0], affs_data[1]], axis=0
            )

        generator = waterz.agglomerate(
            affs=affs_data,
            thresholds=[0, 1.0],
            fragments=frags_relabelled,
            scoring_function=self.merge_function,
            discretize_queue=256,
            return_merge_history=True,
            return_region_graph=True,
        )

        # threshold 0: initial region graph edges
        _, _, initial_rag = next(generator)
        for edge in initial_rag:
            u = int(relabel_map[edge["u"]])
            v = int(relabel_map[edge["v"]])
            rag.add_edge(u, v, merge_score=None)

        # threshold 1.0: full merge history
        _, merge_history, _ = next(generator)
        for _ in generator:
            pass

        merge_tree = MergeTree(relabel_map)
        for merge in merge_history:
            merge_tree.merge(
                relabel_map[merge["a"]],
                relabel_map[merge["b"]],
                relabel_map[merge["c"]],
                merge["score"],
            )

        edge_list = list(rag.edges(data=True))
        if edge_list:
            scores = merge_tree.find_merges(
                [e[0] for e in edge_list], [e[1] for e in edge_list]
            )
            for (u, v, data), s in zip(edge_list, scores):
                data["merge_score"] = None if np.isnan(s) else float(s)

        rag_provider.write_graph(rag, block.write_roi, write_nodes=False)

    @contextmanager
    def process_block_func(self) -> Generator[Callable, None, None]:
        affs = self.affs_data.array("r")
        frags = self.frags_data.array("r")
        rag_provider: GraphDataBase = self.db.open("r+")

        def process_block(block: Block) -> None:
            self.agglomerate_in_block(block, affs, frags, rag_provider)

        yield process_block
