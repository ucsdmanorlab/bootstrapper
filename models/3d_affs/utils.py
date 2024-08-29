import gunpowder as gp
import numpy as np
import random
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt, gaussian_filter, maximum_filter, generate_binary_structure
from skimage.measure import label
from skimage.morphology import disk, star, ellipse
from skimage.segmentation import expand_labels, watershed
from skimage.utils import random_noise


def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):
    method_padding = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (
            gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]])
            + method_padding
        ),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


class SmoothAugment(gp.BatchFilter):
    def __init__(self, array, blur_range):
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