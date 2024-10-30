import logging

import mwatershed as mws
import numpy as np
from scipy.ndimage.filters import gaussian_filter


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mwatershed_from_affinities(
    affs: np.ndarray,
    neighborhood: list[list[int]],
    bias: list[float],
    sigma: list[int] | None = None,
    noise_eps: float | None = None,
    strides: list[list[int]] | None = None,
    randomized_strides: bool = False,
):
    if sigma is not None:
        # add 0 for channel dim
        sigma = (0, *sigma)
    else:
        sigma = None

    # add some random noise to affs (this is particularly necessary if your affs are
    #  stored as uint8 or similar)
    # If you have many affinities of the exact same value the order they are processed
    # in may be fifo, so you can get annoying streaks.

    ### tmp comment out ###

    shift = np.zeros_like(affs)

    if noise_eps is not None:
        shift += np.random.randn(*affs.shape) * noise_eps

    #######################

    # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # towards processing the central regions of objects first.

    ### tmp comment out ###

    if sigma is not None:
        shift += gaussian_filter(affs, sigma=sigma) - affs

    #######################
    shift += np.array([bias]).reshape(
        (-1, *((1,) * (len(affs.shape) - 1)))
    )

    fragments_data = mws.agglom(
        (affs + shift).astype(np.float64),
        offsets=neighborhood,
        strides=strides,
        randomized_strides=randomized_strides,
    )

    return fragments_data

