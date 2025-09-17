import random

import gunpowder as gp
import numpy as np
import edt
from scipy.ndimage import generate_binary_structure, maximum_filter, label
from skimage.morphology import star, disk, ellipse
from skimage.segmentation import watershed

class ObfuscateLabels(gp.BatchFilter):
    """
    Modifies 3D labels arrays by performing random splits, merges, and artifacts.
    Parameters:
        in_labels (str): The name of the array in the batch to modify.
        out_labels (str): The name of the array in the batch to store modified labels.
        p_split (float): The probability of performing a split operation.
        p_merge (float): The probability of performing a merge operation.
        p_artifact (float): The probability of adding artifacts.
    """

    def __init__(
        self,
        in_labels,
        out_labels,
        num_tries=5,
        p_split=0.5,
        p_merge=0.5,
        p_artifact=0.5
    ):
        self.in_labels = in_labels
        self.out_labels = out_labels
        self.num_tries = num_tries
        self.split_p = p_split
        self.merge_p = p_merge
        self.artifact_p = p_artifact

    def setup(self):
        self.enable_autoskip()
        self.provides(self.out_labels, self.spec[self.in_labels])

    def prepare(self, request):
        deps = gp.BatchRequest()

        requested_roi = request[self.out_labels].roi
        deps[self.in_labels] = request[self.in_labels].copy()
        deps[self.in_labels].roi = requested_roi

        return deps

    def process(self, batch, request):

        labels = batch[self.in_labels].data.copy()
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background

        if len(unique_labels) == 0:
            batch[self.out_labels] = gp.Array(labels, batch[self.in_labels].spec)
            return
        
        # pre-generate all random operations
        operations = self._generate_operations()

        for operation in operations:
            if operation == 'split' and len(unique_labels) > 0:
                unique_labels = self._split_labels(labels, unique_labels)

            if operation == 'merge' and len(unique_labels) >= 2:
                unique_labels = self._merge_labels(labels, unique_labels)

            if operation == 'artifact' and len(unique_labels) > 0:
                self._add_artifacts(labels, unique_labels)

        # Update the batch with modified labels
        batch[self.out_labels] = gp.Array(labels, batch[self.in_labels].spec)

    def _generate_operations(self):
        operations = []
        for _ in range(self.num_tries):
            r = random.random()
            if r < self.split_p:
                operations.append('split')
            if r < self.merge_p:
                operations.append('merge')
            if r < self.artifact_p:
                operations.append('artifact')
        return operations

    def _split_labels(self, labels, unique_labels):
        label_to_split = random.choice(unique_labels)
        mask = labels == label_to_split

        dt = edt.edt(mask)
        seeds, _ = label(maximum_filter(dt, size=random.randint(15, 50)) == dt)
        fragments = watershed(dt.max() - dt, seeds, mask=mask) * labels.max()

        # pick some random z slices to apply the split
        z_slices = random.sample(range(mask.shape[0]), k=random.randint(1, 2))

        for z in z_slices:
            slice_mask = mask[z]
            if np.any(slice_mask):
                labels[z] = np.where(slice_mask, fragments[z], labels[z])

        return np.unique(labels[labels != 0])

    def _merge_labels(self, labels, unique_labels):
        # pick some random z slices to apply the merge
        z_slices = random.sample(range(labels.shape[0]), k=random.randint(1, 2))

        labels_to_merge = random.sample(list(unique_labels), 2)

        for z in z_slices:
            labels[z][labels[z] == labels_to_merge[1]] = labels_to_merge[0]

        return unique_labels[unique_labels != labels_to_merge[1]]

    def _add_artifacts(self, labels, unique_labels):

        structs = [
            star(random.randint(2, 8)),
            generate_binary_structure(2, random.randint(1, 2)),
            disk(random.randint(1, 8)),
            ellipse(random.randint(2, 8), random.randint(2, 8)),
        ]

        new_label = labels.max() + 1

        # pick some random z slices to apply the artifacts
        z_slices = random.sample(range(labels.shape[0]), k=random.randint(1, 2))

        for z in z_slices:
            artifact = random.choice(structs)

            # Randomly position the artifact within the slice
            shift_y = random.randint(0, labels.shape[1] - artifact.shape[0])
            shift_x = random.randint(0, labels.shape[2] - artifact.shape[1])

            labels[z, shift_y:shift_y + artifact.shape[0], shift_x:shift_x + artifact.shape[1]] = np.where(
                artifact, new_label, labels[z, shift_y:shift_y + artifact.shape[0], shift_x:shift_x + artifact.shape[1]]
            )
            new_label += 1
        
        return np.unique(labels[labels != 0])