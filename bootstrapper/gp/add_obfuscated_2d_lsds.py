import gunpowder as gp
import numpy as np
import random

from lsd.train.gp import AddLocalShapeDescriptor
from lsd.train import LsdExtractor


class AddObfuscated2DLSDs(AddLocalShapeDescriptor):
    def __init__(self, segmentation, descriptor, *args, **kwargs):

        super().__init__(segmentation, descriptor, *args, **kwargs)

        self.extractor = LsdExtractor(self.sigma[1:], self.mode, self.downsample)

    def process(self, batch, request):

        labels = batch[self.segmentation].data

        spec = batch[self.segmentation].spec.copy()

        spec.dtype = np.float32

        descriptor = np.zeros(shape=(6, *labels.shape))

        for z in range(labels.shape[0]):
            labels_sec = np.copy(labels[z])

            if np.random.random() > 0.2:
                labels_sec = self._random_merge(labels_sec)

            descriptor_sec = self.extractor.get_descriptors(
                segmentation=labels_sec, voxel_size=spec.voxel_size[1:]
            )

            descriptor[:, z] = descriptor_sec

        batch = gp.Batch()
        batch[self.descriptor] = gp.Array(descriptor.astype(spec.dtype), spec)

        return batch

    def _random_merge(self, array, num_pairs_to_merge=4):

        unique_ids = np.unique(array)

        if len(unique_ids) < 2:
            return array

        np.random.shuffle(unique_ids)

        # Determine the number of pairs we can merge
        max_pairs = len(unique_ids) // 2
        pairs_to_merge = min(num_pairs_to_merge, max_pairs)

        for _ in range(random.randrange(pairs_to_merge)):
            label1, label2 = np.random.choice(unique_ids, 2, replace=False)
            array[array == label2] = label1
            unique_ids = unique_ids[unique_ids != label2]

        return array
