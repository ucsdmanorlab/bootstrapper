import random

import gunpowder as gp
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage.morphology import star, disk, ellipse


class ObfuscateAffs(gp.BatchFilter):
    """
    Modifies 2D affinity arrays by creating random blobs of
    positive affinities in areas that were previously negative.
    Parameters:
        affinity_array (str): The name of the array in the batch to modify.
        blob_size_range (tuple): The range of possible blob sizes (min, max) pixels.
        num_blobs_range (tuple): The range for the number of blobs to create (min, max).
        probability (float): The probability of applying the modification to a batch.
    """

    def __init__(
        self,
        affinity_array,
        blob_size_range=(40, 60),
        num_blobs_range=(5, 20),
        blob_dilation_range=(1, 4),
        p=1.0,
    ):
        self.affinity_array = affinity_array
        self.blob_size_range = blob_size_range
        self.num_blobs_range = num_blobs_range
        self.blob_dilation_range = blob_dilation_range
        self.p = p

    def setup(self):
        self.enable_autoskip()
        self.updates(self.affinity_array, self.spec[self.affinity_array])

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):

        affinities = batch[self.affinity_array].data

        # Find boundary regions (where both channels are 0)
        boundary_mask = np.all(affinities == 0, axis=0)

        # Generate random blobs
        num_blobs = np.random.randint(*self.num_blobs_range)
        for _ in range(num_blobs):
            self._create_and_place_blob(affinities, boundary_mask)

        # Update the batch with modified affinities
        batch[self.affinity_array].data = affinities

    def _create_and_place_blob(self, affinities, boundary_mask):

        blob_size = np.random.randint(*self.blob_size_range)
        
        structs = [
            star(random.randint(4, 6)),
            generate_binary_structure(2, 2),
            star(random.randint(3, 5)),
            disk(random.randint(1, 4)),
            star(random.randint(2, 4)),
            ellipse(random.randint(2, 4), random.randint(2, 4)),
            star(random.randint(6, 8)),
        ]

        # Create a random 2D blob
        blob = np.zeros((1, blob_size, blob_size), dtype=bool)
        blob[:,blob_size//2,blob_size//2] = True
        blob[0] = binary_dilation(blob[0], iterations=random.randint(*self.blob_dilation_range), structure=random.choice(structs))

        # Find a random position to place the blob
        valid_positions = np.where(boundary_mask)

        if len(valid_positions[0]) > 0:
            idx = np.random.randint(len(valid_positions[0]))
            z, y, x = (
                valid_positions[0][idx],
                valid_positions[1][idx],
                valid_positions[2][idx],
            )

            # Place the blob
            z_start, y_start, x_start = (
                max(0, z - 1),
                max(0, y - blob_size // 2),
                max(0, x - blob_size // 2),
            )
            z_end, y_end, x_end = (
                min(affinities.shape[1], z_start + 1),
                min(affinities.shape[2], y_start + blob_size),
                min(affinities.shape[3], x_start + blob_size),
            )
            blob_slice = blob[
                : z_end - z_start, : y_end - y_start, : x_end - x_start
            ]

            # Apply the blob to all channels
            for c in range(affinities.shape[0]):
                affinities[c, z_start:z_end, y_start:y_end, x_start:x_end][
                    blob_slice
                ] = 1
