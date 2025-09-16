import random

import gunpowder as gp
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage.morphology import star, disk, ellipse

class ObfuscateLabels(gp.BatchFilter):
    """
    Modifies 3D labels arrays by performing random splits, merges, and artifacts.
    Parameters:
        in_labels (str): The name of the array in the batch to modify.
        out_labels (str): The name of the array in the batch to store modified labels.
        split_p (float): The probability of performing a split operation.
        merge_p (float): The probability of performing a merge operation.
        artifact_p (float): The probability of adding artifacts.
    """

    def __init__(
        self,
        in_labels,
        out_labels,
        num_tries=5,
        split_p=0.5,
        merge_p=0.5,
        artifact_p=0.5
    ):
        self.in_labels = in_labels
        self.out_labels = out_labels
        self.num_tries = num_tries
        self.split_p = split_p
        self.merge_p = merge_p
        self.artifact_p = artifact_p

    def setup(self):
        self.provides(self.out_labels, self.spec[self.in_labels])
        self.enable_autoskip()

    def prepare(self, request):
        deps = gp.BatchRequest()

        requested_roi = request[self.out_labels].roi
        deps[self.in_labels] = request[self.in_labels].copy()
        deps[self.in_labels].roi = requested_roi

        return deps

    def process(self, batch, request):

        labels = batch[self.in_labels].data.copy()

        for _ in range(self.num_tries):
            if random.random() < self.split_p:
                self._split_labels(labels)

            if random.random() < self.merge_p:
                self._merge_labels(labels)

            if random.random() < self.artifact_p:
                self._add_artifacts(labels)

        # Update the batch with modified labels
        batch[self.out_labels] = gp.Array(labels, batch[self.in_labels].spec)

    def _split_labels(self, labels):
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background

        if len(unique_labels) == 0:
            return

        label_to_split = random.choice(unique_labels)
        mask = labels == label_to_split
        dilated_mask = np.zeros_like(mask, dtype=bool)

        structs = [
            star(random.randint(2, 8)),
            generate_binary_structure(2, random.randint(1,2)),
            disk(random.randint(1, 8)),
            ellipse(random.randint(2, 8), random.randint(2, 8)),
        ]

        # Create a random split using a binary structure
        struct = random.choice(structs)
        for z in range(mask.shape[0]):
            dilated_mask[z] = binary_dilation(mask[z], structure=struct, iterations=random.randint(2, 5))

        new_label = labels.max() + 1

        # pick some random z slices to apply the split
        z_slices = random.sample(range(mask.shape[0]), k=random.randint(1, 3))

        for z in z_slices:
            labels[z][dilated_mask[z] & ~mask[z]] = new_label
            new_label += 1

    def _merge_labels(self, labels):
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background

        if len(unique_labels) < 2:
            return

        # pick some random z slices to apply the merge
        z_slices = random.sample(range(labels.shape[0]), k=random.randint(1, 3))

        labels_to_merge = random.sample(list(unique_labels), 2)

        for z in z_slices:
            labels[z][labels[z] == labels_to_merge[1]] = labels_to_merge[0]

    def _add_artifacts(self, labels):
        num_artifacts = random.randint(1, 4)

        structs = [
            star(random.randint(2, 8)),
            generate_binary_structure(2, random.randint(1, 2)),
            disk(random.randint(1, 8)),
            ellipse(random.randint(2, 8), random.randint(2, 8)),
        ]

        new_label = labels.max() + 1

        # pick some random z slices to apply the artifacts
        z_slices = random.sample(range(labels.shape[0]), k=random.randint(1, 3))

        for z in z_slices:
            artifact = random.choice(structs)

            # Randomly position the artifact within the slice
            shift_y = random.randint(0, labels.shape[1] - artifact.shape[0])
            shift_x = random.randint(0, labels.shape[2] - artifact.shape[1])
            artifact = np.pad(artifact, ((shift_y, labels.shape[1] - artifact.shape[0] - shift_y), (shift_x, labels.shape[2] - artifact.shape[1] - shift_x)))

            labels[z][artifact] = new_label
            new_label += 1