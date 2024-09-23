import gunpowder as gp
import numpy as np
import random
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
    maximum_filter,
    gaussian_filter,
)
from skimage.morphology import (
    disk,
    ellipse,
    star,
)
from skimage.segmentation import watershed
from skimage.measure import label


class CreateLabels(gp.BatchProvider):
    """
    A provider node for generating synthetic 3D labels arrays.

    Args:
        array_key (gp.ArrayKey): The key of the array to provide labels for.
        anisotropy_range (tuple): The range of anisotropy values to use for label generation.
        shape (tuple): The shape of the labels array.
        dtype (numpy.dtype): The data type of the labels.
        voxel_size (tuple): The voxel size of the labels.
    """

    def __init__(
        self,
        array_key,
        anisotropy_range=None,
        shape=(20, 20, 20),
        dtype=np.uint32,
        voxel_size=None,
    ):
        self.array_key = array_key
        self.anisotropy_range = anisotropy_range
        self.shape = shape
        self.dtype = dtype
        self.voxel_size = voxel_size
        self.ndims = None

    def setup(self):
        spec = gp.ArraySpec()

        if self.voxel_size is None:
            voxel_size = gp.Coordinate((1,) * len(self.shape))
        else:
            voxel_size = gp.Coordinate(self.voxel_size)

        spec.voxel_size = voxel_size
        self.ndims = len(spec.voxel_size)

        if self.anisotropy_range is None:
            self.anisotropy_range = (4, int(voxel_size[0] / voxel_size[1]))

        offset = gp.Coordinate((0,) * self.ndims)
        spec.roi = gp.Roi(offset, gp.Coordinate(self.shape) * spec.voxel_size)
        spec.dtype = self.dtype
        spec.interpolatable = False

        self.provides(self.array_key, spec)

    def provide(self, request):
        batch = gp.Batch()

        request_spec = request.array_specs[self.array_key]
        voxel_size = self.spec[self.array_key].voxel_size

        # scale request roi to voxel units
        dataset_roi = request_spec.roi / voxel_size

        # shift request roi into dataset
        dataset_roi = (
            dataset_roi - self.spec[self.array_key].roi.get_offset() / voxel_size
        )

        # create array spec
        array_spec = self.spec[self.array_key].copy()
        array_spec.roi = request_spec.roi

        labels = self._generate_labels(dataset_roi.to_slices())

        batch.arrays[self.array_key] = gp.Array(labels, array_spec)

        return batch

    def _generate_labels(self, slices):
        shape = tuple(s.stop - s.start for s in slices)
        labels = np.zeros(shape, self.dtype)
        anisotropy = random.randint(*self.anisotropy_range)
        labels = np.concatenate([labels] * anisotropy)
        shape = labels.shape

        choice = random.choice(["tubes", "random"])

        if choice == "tubes":
            num_points = random.randint(5, 5 * anisotropy)
            for n in range(num_points):
                z = random.randint(1, labels.shape[0] - 1)
                y = random.randint(1, labels.shape[1] - 1)
                x = random.randint(1, labels.shape[2] - 1)
                labels[z, y, x] = 1

            for z in range(labels.shape[0]):
                dilations = random.randint(1, 10)
                structs = [
                    generate_binary_structure(2, 2),
                    disk(random.randint(1, 4)),
                    star(random.randint(2, 4)),
                    ellipse(random.randint(2, 4), random.randint(2, 4)),
                ]
                dilated = binary_dilation(
                    labels[z], structure=random.choice(structs), iterations=dilations
                )
                labels[z] = dilated.astype(labels.dtype)

            labels = label(labels)

            distance = labels.shape[0]
            distances, indices = distance_transform_edt(
                labels == 0, return_indices=True
            )
            expanded_labels = np.zeros_like(labels)
            dilate_mask = distances <= distance
            masked_indices = [
                dimension_indices[dilate_mask] for dimension_indices in indices
            ]
            nearest_labels = labels[tuple(masked_indices)]
            expanded_labels[dilate_mask] = nearest_labels
            labels = expanded_labels

            labels[labels == 0] = np.max(labels) + 1
            labels = label(labels)[::anisotropy].astype(np.uint32)

        elif choice == "random":
            np.random.seed()
            peaks = np.random.random(shape).astype(np.float32)
            peaks = gaussian_filter(peaks, sigma=10.0)
            max_filtered = maximum_filter(peaks, 15)
            maxima = max_filtered == peaks
            seeds = label(maxima, connectivity=1)
            labels = watershed(1.0 - peaks, seeds)[::anisotropy].astype(np.uint32)

        else:
            raise AssertionError("invalid choice")

        return labels
