from contextlib import contextmanager

import mwatershed as mws
import numpy as np
from funlib.geometry import Coordinate
from funlib.persistence import Array
from scipy.ndimage import center_of_mass, gaussian_filter
from scipy.ndimage import maximum as ndi_maximum
from scipy.ndimage import mean as ndi_mean
from skimage.measure import label as relabel
from skimage.morphology import remove_small_objects

from volara.blockwise import ExtractFrags as VolaraExtractFrags
from volara.datasets import Labels
from volara.tmp import replace_values


class ExtractFrags(VolaraExtractFrags):
    """ExtractFrags with the fragment filter written back into the array.

    ``replace_values`` returns a new array, so the parent's filter result is lost.
    """

    def filter_avg_fragments(self, affs, fragments_data, filter_value):
        average_affs = np.mean(affs[0:3], axis=0)
        fragment_ids = np.unique(fragments_data)
        means = ndi_mean(average_affs, fragments_data, fragment_ids)
        filtered = np.array(
            [f for f, m in zip(fragment_ids, means) if m < filter_value],
            dtype=fragments_data.dtype,
        )
        fragments_data[:] = replace_values(
            fragments_data, filtered, np.zeros_like(filtered)
        )


class SeededExtractFrags(ExtractFrags):
    """ExtractFrags seeded from an integer labels array on the affs grid.

    mwatershed keeps every nonzero seed voxel in its seed's fragment and never
    joins two seed ids. Seeded fragments skip the fragment filters. The db node
    attribute seed_id is the id of the seed voxels a fragment holds, else 0, as text.
    """

    seeds_data: Labels

    def compute_fragments(self, affs_data, seeds):
        shift = np.zeros_like(affs_data)
        if self.noise_eps is not None:
            shift += np.random.randn(*affs_data.shape) * self.noise_eps
        if self.sigma is not None:
            shift += gaussian_filter(affs_data, sigma=(0, *self.sigma)) - affs_data
        shift += np.array([self.bias]).reshape(
            (-1, *((1,) * (len(affs_data.shape) - 1)))
        )
        return mws.agglom(
            (affs_data + shift).astype(np.float64),
            offsets=self.neighborhood,
            strides=self.strides,
            seeds=seeds,
            randomized_strides=self.randomized_strides,
        )

    def get_fragments(self, affs_data, seeds):
        """Fragments where a seeded fragment is its seed id + seeds.size.

        Unseeded mwatershed ids are at most the voxel count, so the shift keeps
        the two apart."""
        n = np.uint64(seeds.size)
        fragments_data = self.compute_fragments(
            affs_data, np.where(seeds > 0, seeds + n, 0).astype(np.uint64)
        )
        seeded = fragments_data > n
        seeded_ids = fragments_data[seeded]
        fragments_data[seeded] = 0

        if self.filter_fragments > 0:
            self.filter_avg_fragments(affs_data, fragments_data, self.filter_fragments)
        if self.remove_debris > 0:
            fragments_dtype = fragments_data.dtype
            fragments_data = remove_small_objects(
                fragments_data.astype(np.int64), min_size=self.remove_debris
            ).astype(fragments_dtype)

        fragments_data[seeded] = seeded_ids
        return fragments_data

    def watershed_in_block(self, block, affs, frags, rag_provider, mask=None, seeds=None):
        affs_data = affs.to_ndarray(block.read_roi, fill_value=0)
        seeds_data = seeds.to_ndarray(block.read_roi, fill_value=0).astype(np.uint64)

        if affs.dtype == np.uint8:
            affs_data = affs_data.astype(np.float64) / 255.0
        # a block with no signal is empty unless it holds seeds
        if affs_data.max() < 1e-3 and not seeds_data.any():
            return

        if mask is not None:
            mask_data = mask.to_ndarray(block.read_roi, fill_value=0)
            if len(mask_data.shape) == block.read_roi.dims + 1:
                mask_data = (np.min(mask_data, axis=0) > 0).astype(np.uint8)
            if np.max(mask_data) == 255:
                mask_data = (mask_data > 0).astype(np.uint8)
            affs_data *= mask_data

        fragments_data = self.get_fragments(affs_data, seeds_data)
        fragments_data, seeds_data = (
            Array(
                data, offset=block.read_roi.offset, voxel_size=frags.voxel_size
            ).to_ndarray(block.write_roi)
            for data in (fragments_data, seeds_data)
        )

        labels, max_id = relabel(fragments_data, return_num=True)
        assert max_id < self.num_voxels_in_block, f"max_id: {max_id}"
        id_bump = block.block_id[1] * self.num_voxels_in_block
        labels[labels > 0] += id_bump
        frags[block.write_roi] = labels

        if max_id == 0:
            return

        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        centers = center_of_mass(np.ones_like(labels), labels, ids)
        # seeds are exclusive, so a fragment holds voxels of one seed id at most
        seed_ids = ndi_maximum(seeds_data, labels, ids)
        rag = rag_provider[block.write_roi]
        for fragment_id, center, count, seed_id in zip(ids, centers, counts, seed_ids):
            rag.add_node(
                int(fragment_id),
                position=block.write_roi.get_offset()
                + affs.voxel_size * Coordinate(center),
                size=int(count),
                seed_id=str(seed_id),
            )
        rag_provider.write_graph(rag, block.write_roi)

    @contextmanager
    def process_block_func(self):
        affs = self.affs_data.array("r")
        frags = self.frags_data.array("r+")
        mask = self.mask_data.array("r") if self.mask_data else None
        seeds = self.seeds_data.array("r")
        rag_provider = self.db.open("r+")

        def process_block(block):
            self.watershed_in_block(
                block, affs, frags, rag_provider, mask=mask, seeds=seeds
            )

        yield process_block
