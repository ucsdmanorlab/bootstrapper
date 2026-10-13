"""filter_fragments must change the fragments, on both segmentation paths.

volara's ``replace_values`` is njit compiled: it returns a new array and edits
nothing in place. Both fragment tasks used to drop that return value, so
``filter_fragments`` was a no-op and every ``ff*`` dataset name lied.
"""

import numpy as np
from volara.datasets import Affs, Labels, Raw
from volara.dbs import SQLite

from bootstrapper.post.blockwise.extract_frags import ExtractFrags
from bootstrapper.post.blockwise.watershed_frags import WatershedFrags

FILTER_VALUE = 0.5


def fragments():
    """Two fragments: 1 over low affinities, 2 over high affinities."""
    frags = np.zeros((2, 4, 4), dtype=np.uint64)
    frags[:, :2, :] = 1
    frags[:, 2:, :] = 2
    return frags


def affinities():
    affs = np.zeros((3, 2, 4, 4), dtype=np.float64)
    affs[:, :, :2, :] = 0.1
    affs[:, :, 2:, :] = 0.9
    return affs


def watershed_task(**kwargs):
    class Task(WatershedFrags):
        def compute_fragments(self, affs_data):
            return fragments()

    return Task(
        db=SQLite(path="rag.db"),
        affs_data=Raw(store="test.zarr/affs"),
        frags_data=Labels(store="test.zarr/frags"),
        block_size=(2, 4, 4),
        context=(0, 0, 0),
        **kwargs,
    )


def extract_task(**kwargs):
    class Task(ExtractFrags):
        def compute_fragments(self, affs_data):
            return fragments()

    return Task(
        db=SQLite(path="rag.db"),
        affs_data=Affs(
            store="test.zarr/affs", neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        ),
        frags_data=Labels(store="test.zarr/frags"),
        block_size=(2, 4, 4),
        context=(0, 0, 0),
        bias=[-0.5, -0.5, -0.5],
        **kwargs,
    )


def test_watershed_filter_returns_the_filtered_fragments():
    task = watershed_task(filter_fragments=FILTER_VALUE)

    filtered = task.filter_avg_fragments(affinities(), fragments(), FILTER_VALUE)

    assert list(np.unique(filtered)) == [0, 2]


def test_watershed_fragments_drop_the_filtered_fragment():
    task = watershed_task(filter_fragments=FILTER_VALUE)

    assert list(np.unique(task.get_fragments(affinities()))) == [0, 2]


def test_mutex_fragments_drop_the_filtered_fragment():
    task = extract_task(filter_fragments=FILTER_VALUE)

    assert list(np.unique(task.get_fragments(affinities()))) == [0, 2]
