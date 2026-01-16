import gunpowder as gp
import numpy as np
import random
from scipy.ndimage import gaussian_filter
import itertools


class SmoothAugment(gp.BatchFilter):
    """Randomly scale and shift the values of an intensity array.

    Args:

        array (:class:`ArrayKey`):

            The intensity array to modify.

        blur_min (``float``):
        blur_max (``float``):

            The min and max of the uniformly randomly drawn sigma value.
        slab (``tuple`` of ``int``, optional):
            A shape specification to perform the gamma augment in slabs of this
            size. -1 can be used to refer to the actual size of the array.
            For example, a slab of::

                (2, -1, -1, -1)

            will perform the gamma augment for every each slice ``[0:2,:]``,
            ``[2:4,:]``, ... individually on 4D data.
    """

    def __init__(self, array, blur_min=0.5, blur_max=1.5, slab=None, p=1.0):
        self.array = array
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.slab = slab
        self.p = p
        assert self.blur_max >= self.blur_min

    def setup(self):
        self.enable_autoskip()
        self.updates(self.array, self.spec[self.array])

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):

        # array = batch[self.array].data

        # # different numbers will simulate noisier or cleaner array
        # if len(array.shape) == 3:
        #     for z in range(array.shape[0]):
        #         sigma = random.uniform(self.blur_min, self.blur_max)
        #         array_sec = array[z]

        #         array[z] = np.array(gaussian_filter(array_sec, sigma=sigma)).astype(
        #             array_sec.dtype
        #         )

        # elif len(array.shape) == 4:
        #     for z in range(array.shape[1]):
        #         sigma = random.uniform(self.blur_min, self.blur_max)
        #         array_sec = array[:, z]

        #         array[:, z] = np.array(
        #             [
        #                 gaussian_filter(array_sec[i], sigma=sigma)
        #                 for i in range(array_sec.shape[0])
        #             ]
        #         ).astype(array_sec.dtype)

        # elif len(array.shape) == 2:
        #     sigma = random.uniform(self.blur_min, self.blur_max)
        #     array = np.array(gaussian_filter(array, sigma=sigma)).astype(array.dtype)

        # else:
        #     raise AssertionError("array shape is not 2d, 3d, or multi-channel 3d")

        # batch[self.array].data = array

        array_data = batch[self.array].data

        if self.slab is not None:
            slab = self.slab
        else:
            slab = [-1] * len(array_data.shape)

        # slab with -1 replaced by actual shape
        slab = tuple(m if s == -1 else s for m, s in zip(array_data.shape, slab))

        slab_ranges = (range(0, m, s) for m, s in zip(array_data.shape, slab))

        for start in itertools.product(*slab_ranges):
            slices = tuple(
                slice(start[d], start[d] + slab[d]) for d in range(len(slab))
            )
            sigma = random.uniform(self.blur_min, self.blur_max)
            array_data[slices] = gaussian_filter(
                array_data[slices], sigma=sigma
            ).astype(array_data.dtype)

        batch[self.array].data = array_data
