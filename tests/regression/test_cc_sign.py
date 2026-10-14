"""Connected components reads affinities in the convention bs trains them: offset [-1, 0, 0] links a voxel
with the voxel before it, so the edge between x and x + 1 sits at x + 1."""
import numpy as np
from gunpowder.nodes.add_affinities import seg_to_affgraph

from bootstrapper.post.cc import compute_connected_component_segmentation

NEIGHBORHOOD = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])


def same_partition(a, b):
    pairs = set(zip(a.ravel().tolist(), b.ravel().tolist()))
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))


def test_touching_objects_come_back_as_they_were_labelled():
    labels = np.zeros((4, 6, 12), np.uint64)
    labels[:, :, 0:3], labels[:, :, 3:7], labels[:, :, 7:12] = 1, 2, 3
    labels[:, 3:, 3:7] = 4
    labels[2:, :, 7:12] = 5
    affs = seg_to_affgraph(labels, NEIGHBORHOOD) > 0.5

    seg = compute_connected_component_segmentation(affs)

    assert same_partition(seg, labels), "a boundary moved by one voxel or two touching objects joined"


def test_one_line_of_three_objects():
    labels = np.array([1, 1, 2, 2, 2, 3], np.uint64).reshape(6, 1, 1)
    affs = seg_to_affgraph(labels, NEIGHBORHOOD) > 0.5
    affs[1:] = True  # one voxel wide in y and x: those channels carry no boundary

    seg = compute_connected_component_segmentation(affs).ravel()

    assert seg[0] == seg[1] != seg[2] == seg[3] == seg[4] != seg[5]
