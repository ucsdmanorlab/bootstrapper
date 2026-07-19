import logging
from contextlib import contextmanager
from typing import Annotated, Callable, Generator, Literal

import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence.arrays import Array
from pydantic import Field
from scipy.ndimage import (
    center_of_mass,
    distance_transform_edt,
    gaussian_filter,
    label,
    maximum_filter,
)
from scipy.ndimage import mean as ndi_mean
from skimage.measure import label as relabel
from skimage.morphology import remove_small_objects

from volara.blockwise import BlockwiseTask
from volara.datasets import Dataset, Labels, Raw
from volara.dbs import PostgreSQL, SQLite
from volara.tmp import replace_values
from volara.utils import PydanticCoordinate

from ..ws import watershed_from_affinities

logger = logging.getLogger(__name__)


class WatershedFrags(BlockwiseTask):
    """
    A blockwise task that extracts fragments from affinities with a seeded
    watershed (the same fragmentation as the non-blockwise ``simple_watershed``)
    and writes fragment supervoxel nodes (position, size) into the volara graph
    database (SQLite or PostgreSQL) for downstream waterz agglomeration.
    """

    task_type: Literal["watershed-frags"] = "watershed-frags"
    db: Annotated[
        PostgreSQL | SQLite,
        Field(discriminator="db_type"),
    ]
    affs_data: Raw
    frags_data: Labels
    mask_data: Raw | None = None
    block_size: PydanticCoordinate
    context: PydanticCoordinate
    fragments_in_xy: bool = True
    min_seed_distance: int = 10
    seed_eps: float | None = None
    epsilon_agglomerate: float = 0.0
    sigma: PydanticCoordinate | None = None
    noise_eps: float | None = None
    bias: list[float] | float | None = None
    filter_fragments: float = 0.0
    remove_debris: int = 0

    fit: Literal["shrink"] = "shrink"
    read_write_conflict: Literal[False] = False
    _out_array_dtype: np.dtype = np.dtype(np.uint64)

    @property
    def task_name(self) -> str:
        return f"{self.frags_data.name}-{self.task_type}"

    @property
    def write_roi(self) -> Roi:
        total_roi = self.affs_data.array("r").roi
        if self.roi is not None:
            total_roi = total_roi.intersect(self.roi)
        return total_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.affs_data.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.block_size * self.voxel_size

    @property
    def context_size(self) -> Coordinate:
        return self.context * self.voxel_size

    @property
    def num_voxels_in_block(self) -> int:
        return int(np.prod(self.block_size))

    @property
    def output_datasets(self) -> list[Dataset]:
        return [self.frags_data]

    def drop_artifacts(self):
        self.frags_data.drop()
        self.db.drop()

    def init(self):
        self.db.init()
        self.init_out_array()

    def init_out_array(self):
        in_data = self.affs_data.array("r")
        self.frags_data.prepare(
            self.write_roi.shape / self.voxel_size,
            self.write_size / self.voxel_size,
            self.write_roi.offset,
            self.voxel_size,
            units=in_data.units,
            axis_names=in_data.axis_names[1:],
            types=in_data.types[1:],
            dtype=self._out_array_dtype,
        )

    def compute_fragments(self, affs_data):
        # watershed only uses the first 3 (z, y, x) nearest-neighbor affinities
        affs_data = affs_data[:3]
        shift = np.zeros_like(affs_data)
        if self.noise_eps is not None:
            shift += np.random.randn(*affs_data.shape) * self.noise_eps
        if self.sigma is not None:
            shift += gaussian_filter(affs_data, sigma=(0, *self.sigma)) - affs_data
        if self.bias is not None:
            bias = (
                list(self.bias)
                if isinstance(self.bias, (list, tuple))
                else [self.bias] * affs_data.shape[0]
            )
            shift += np.array([bias]).reshape((-1, *((1,) * (len(affs_data.shape) - 1))))

        if self.seed_eps is not None:
            # volara ExtractFrags semantics: decay the affs by distance from the
            # seeds to increase fragmentation (shift -= seed_eps * D).
            boundary_mask = np.mean(affs_data, axis=0) > 0.5
            boundary_distances = distance_transform_edt(boundary_mask)
            max_filtered = maximum_filter(boundary_distances, self.min_seed_distance)
            seeds, _ = label(max_filtered == boundary_distances)
            seeds[~boundary_mask] = 0
            shift -= self.seed_eps * distance_transform_edt(seeds == 0)

        fragments_data, _ = watershed_from_affinities(
            affs_data + shift,
            fragments_in_xy=self.fragments_in_xy,
            min_seed_distance=self.min_seed_distance,
        )
        return fragments_data

    def filter_avg_fragments(self, affs, fragments_data, filter_value):
        average_affs = np.mean(affs[0:3], axis=0)
        fragment_ids = np.unique(fragments_data)
        means = ndi_mean(average_affs, fragments_data, fragment_ids)
        filtered = np.array(
            [f for f, m in zip(fragment_ids, means) if m < filter_value],
            dtype=fragments_data.dtype,
        )
        replace_values(fragments_data, filtered, np.zeros_like(filtered))

    def epsilon_agglomerate_fragments(self, affs_data, fragments_data):
        # quick initial waterz merge of the watershed fragments up to a low
        # threshold (lsd uses hist_quant_25 here, but the ZettaAI/waterz build
        # only ships a working "mean" scorer, so we use that).
        import waterz

        affs = np.ascontiguousarray(affs_data[:3].astype(np.float32))
        generator = waterz.agglomerate(
            affs=affs,
            thresholds=[self.epsilon_agglomerate],
            fragments=fragments_data,
            scoring_function="OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
            discretize_queue=256,
            return_merge_history=False,
            return_region_graph=False,
        )
        fragments_data[:] = next(generator)
        for _ in generator:
            pass
        return fragments_data

    def get_fragments(self, affs_data):
        fragments_data = self.compute_fragments(affs_data)

        if self.epsilon_agglomerate > 0:
            fragments_data = self.epsilon_agglomerate_fragments(affs_data, fragments_data)

        if self.filter_fragments > 0:
            self.filter_avg_fragments(affs_data, fragments_data, self.filter_fragments)

        if self.remove_debris > 0:
            dtype = fragments_data.dtype
            fragments_data = remove_small_objects(
                fragments_data.astype(np.int64), min_size=self.remove_debris
            ).astype(dtype)

        return fragments_data

    def watershed_in_block(self, block, affs, frags, rag_provider, mask=None):
        affs_data = affs.to_ndarray(block.read_roi, fill_value=0)
        if affs.dtype == np.uint8:
            max_affinity_value = 255.0
            affs_data = affs_data.astype(np.float64)
        else:
            max_affinity_value = 1.0
        if affs_data.max() < 1e-3:
            return
        affs_data /= max_affinity_value

        if mask is not None:
            mask_data = mask.to_ndarray(block.read_roi, fill_value=0)
            if len(mask_data.shape) == block.read_roi.dims + 1:
                mask_data = (np.min(mask_data, axis=0) > 0).astype(np.uint8)
            if np.max(mask_data) == 255:
                mask_data = (mask_data > 0).astype(np.uint8)
            affs_data *= mask_data

        fragments_data = self.get_fragments(affs_data)
        fragments = Array(
            fragments_data, offset=block.read_roi.offset, voxel_size=frags.voxel_size
        )

        # crop to the write roi and give every fragment a globally-unique id
        fragments_data = fragments.to_ndarray(block.write_roi)
        fragments_data, max_id = relabel(fragments_data, return_num=True)
        assert max_id < self.num_voxels_in_block, f"max_id: {max_id}"
        fragments_data[fragments_data > 0] += block.block_id[1] * self.num_voxels_in_block

        frags[block.write_roi] = fragments_data
        if fragments_data.max() == 0:
            return

        fragment_ids, counts = np.unique(fragments_data, return_counts=True)
        fragment_ids, counts = zip(
            *[(f, c) for f, c in zip(fragment_ids, counts) if f > 0]
        )
        logger.info("Found %d fragments", len(fragment_ids))
        centers = center_of_mass(
            np.ones_like(fragments_data), fragments_data, list(fragment_ids)
        )

        rag = rag_provider[block.write_roi]
        for fid, center, count in zip(fragment_ids, centers, counts):
            rag.add_node(
                int(fid),
                position=block.write_roi.offset + self.voxel_size * Coordinate(center),
                size=int(count),
            )
        rag_provider.write_graph(rag, block.write_roi)

    @contextmanager
    def process_block_func(self) -> Generator[Callable, None, None]:
        affs = self.affs_data.array("r")
        frags = self.frags_data.array("r+")
        mask = self.mask_data.array("r") if self.mask_data else None
        rag_provider = self.db.open("r+")

        def process_block(block) -> None:
            self.watershed_in_block(block, affs, frags, rag_provider, mask=mask)

        yield process_block
