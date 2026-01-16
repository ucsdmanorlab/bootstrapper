import gunpowder as gp
import numpy as np
from scipy.ndimage import distance_transform_edt


class ExpandLabels(gp.BatchFilter):
    def __init__(self, labels, background=0, expansion_factor=1):
        self.labels = labels
        self.background = (background,)
        self.expansion_factor = expansion_factor

    def process(self, batch, request):
        labels = batch[self.labels].data
        expanded_labels = np.zeros_like(labels)

        z_slices = labels.shape[0]

        for z in range(z_slices):
            z_slice = labels[z]

            distances, indices = distance_transform_edt(
                z_slice == self.background, return_indices=True
            )

            dilate_mask = distances <= self.expansion_factor
            masked_indices = [
                dimension_indices[dilate_mask] for dimension_indices in indices
            ]
            nearest_labels = z_slice[tuple(masked_indices)]

            expanded_labels[z][dilate_mask] = nearest_labels

        batch[self.labels].data = expanded_labels