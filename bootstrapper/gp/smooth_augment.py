import gunpowder as gp
import numpy as np
import random
from scipy.ndimage import gaussian_filter


class SmoothAugment(gp.BatchFilter):
    """Randomly scale and shift the values of an intensity array.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify.

        blur_min (``float``):
        blur_max (``float``):

            The min and max of the uniformly randomly drawn sigma value.
    """

    def __init__(self, array, blur_min=0.75, blur_max=1.25, p=1.0):
        self.array = array
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.p = p

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):

        array = batch[self.array].data

        # different numbers will simulate noisier or cleaner array
        if len(array.shape) == 3:
            for z in range(array.shape[0]):
                sigma = random.uniform(self.blur_min, self.blur_max)
                array_sec = array[z]

                array[z] = np.array(gaussian_filter(array_sec, sigma=sigma)).astype(
                    array_sec.dtype
                )

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
            array = np.array(gaussian_filter(array, sigma=sigma)).astype(array.dtype)

        else:
            raise AssertionError("array shape is not 2d, 3d, or multi-channel 3d")

        batch[self.array].data = array
