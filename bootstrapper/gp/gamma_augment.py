# adapted from https://github.com/saalfeldlab/corditea/blob/main/src/corditea/gamma_augment.py

import itertools
import logging
from collections.abc import Iterable

import numpy as np
from gunpowder import BatchFilter

logger = logging.getLogger(__name__)


class GammaAugment(BatchFilter):
    """
    An Augment to apply gamma noise

    Parameters:
        arrays (:class:`ArrayKey` or list of :class:`ArrayKey`s):
            The intensity arrays to modify, applying the same noise pattern to each..
        gamma_min (``float``):
            Minimum gamma value to sample from.
        gamma_max (``float``):
            Maximum gamma value to sample from. Must be >= gamma_min.
        slab (``tuple`` of ``int``, optional):
            A shape specification to perform the gamma augment in slabs of this
            size. -1 can be used to refer to the actual size of the array.
            For example, a slab of::

                (2, -1, -1, -1)

            will perform the gamma augment for every each slice ``[0:2,:]``,
            ``[2:4,:]``, ... individually on 4D data.
        p (``float``, optional):
            Probability with which to apply this augmentation.
    """

    def __init__(self, arrays, gamma_min=0.8, gamma_max=1.2, slab=None, p=1.0):
        if not isinstance(arrays, Iterable):
            arrays = [
                arrays,
            ]
        self.arrays = arrays
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.slab = slab
        self.p = p
        assert self.gamma_max >= self.gamma_min

    def setup(self):
        self.enable_autoskip()
        for array in self.arrays:
            self.updates(array, self.spec[array])

    def skip_node(self, request):
        return np.random.random() > self.p

    def process(self, batch, request):
        sample_gamma_min = (max(self.gamma_min, 1.0 / self.gamma_min) - 1) * (-1) ** (self.gamma_min < 1)
        sample_gamma_max = (max(self.gamma_max, 1.0 / self.gamma_max) - 1) * (-1) ** (self.gamma_max < 1)

        for array in self.arrays:
            raw = batch.arrays[array]

            assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
                "Gamma augmentation requires float "
                "types for the raw array (not " + str(raw.data.dtype) + "). Consider using Normalize before."
            )

            if self.slab is not None:
                slab = self.slab
            else:
                slab = [-1] * len(raw.data.shape)

            # slab with -1 replaced by shape
            slab = tuple(m if s == -1 else s for m, s in zip(raw.data.shape, slab))

            slab_ranges = (range(0, m, s) for m, s in zip(raw.data.shape, slab))

            for start in itertools.product(*slab_ranges):
                gamma = np.random.uniform(sample_gamma_min, sample_gamma_max)
                if gamma < 0:
                    gamma = 1.0 / (-gamma + 1)
                else:
                    gamma = gamma + 1

                slices = tuple(
                    slice(start[d], start[d] + slab[d]) for d in range(len(slab))
                )
                raw.data[slices] = self.__augment(raw.data[slices], gamma)


    def __augment(self, a, gamma):
        # normalize a
        a_min = a.min()
        a_max = a.max()
        if abs(a_min - a_max) > 1e-3:
            # apply gamma noise
            a = (a - a_min) / (a_max - a_min)
            noisy_a = a**gamma
            # undo normalization
            noisy_a = a * (a_max - a_min) + a_min
            return noisy_a
        else:
            logger.debug("Skipping gamma noise since denominator would be too small")
            return a
