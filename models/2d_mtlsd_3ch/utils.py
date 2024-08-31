import gunpowder as gp
import numpy as np
import random
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter

from lsd.train.gp import AddLocalShapeDescriptor
from lsd.train import LsdExtractor


class SmoothAugment(gp.BatchFilter):
    def __init__(self, array, blur_range=(0.0, 1.0)):
        self.array = array
        self.range = blur_range

    def process(self, batch, request):

        array = batch[self.array].data

        # different numbers will simulate noisier or cleaner array

        if len(array.shape) == 3:
            for z in range(array.shape[0]):
                sigma = random.uniform(self.range[0], self.range[1])
                array_sec = array[z]

                array[z] = np.array(
                        gaussian_filter(array_sec, sigma=sigma)
                ).astype(array_sec.dtype)
        
        elif len(array.shape) == 4:
            for z in range(array.shape[1]):
                sigma = random.uniform(self.range[0], self.range[1])
                array_sec = array[:, z]

                array[:, z] = np.array(
                    [
                        gaussian_filter(array_sec[i], sigma=sigma)
                        for i in range(array_sec.shape[0])
                    ]
                ).astype(array_sec.dtype)
        
        elif len(array.shape) == 2:                
            sigma = random.uniform(self.range[0], self.range[1])
            array = np.array(
                        gaussian_filter(array, sigma=sigma)
            ).astype(array.dtype)

        else:
            raise AssertionError("array shape is not 2d, 3d, or multi-channel 3d")

        batch[self.array].data = array


class CustomLSDs(AddLocalShapeDescriptor):
    def __init__(self, segmentation, descriptor, *args, **kwargs):

        super().__init__(segmentation, descriptor, *args, **kwargs)

        self.extractor = LsdExtractor(
                self.sigma[1:], self.mode, self.downsample
        )

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
            raise ValueError("Not enough unique_ids to merge.")

        np.random.shuffle(unique_ids)

        # Determine the number of pairs we can merge
        max_pairs = len(unique_ids) // 2
        pairs_to_merge = min(num_pairs_to_merge, max_pairs)

        for _ in range(random.randrange(pairs_to_merge)):
            label1, label2 = np.random.choice(unique_ids, 2, replace=False)
            array[array == label2] = label1
            unique_ids = unique_ids[unique_ids != label2]

        return array


class NoiseAugment(gp.BatchFilter):
    def __init__(self, array, mode="gaussian", p=0.5, clip=True, **kwargs):
        self.array = array
        self.mode = mode
        self.clip = clip
        self.kwargs = kwargs
        self.p = p

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.array] = request[self.array].copy()
        return deps

    def process(self, batch, request):

        if np.random.random() > self.p:
            return

        raw = batch.arrays[self.array]

        assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
            "Noise augmentation requires float types for the raw array (not "
            + str(raw.data.dtype)
            + "). Consider using Normalize before."
        )
        if self.clip:
            assert (
                raw.data.min() >= -1 and raw.data.max() <= 1
            ), "Noise augmentation expects raw values in [-1,1] or [0,1]. Consider using Normalize before."

        seed = request.random_seed

        raw.data = random_noise(
            raw.data, mode=self.mode, rng=seed, clip=self.clip, **self.kwargs
        ).astype(raw.data.dtype)
