import logging

import mwatershed as mws
import numpy as np
from scipy.ndimage.filters import gaussian_filter

logging.basic(level=logging.INFO)



def mwatershed_from_affinities(
    affs,
    neighborhood,
    sigma,
    adjacent_edge_bias,
    lr_edge_bias,
    strides=None,
):

    # add some random noise to affs (this is particularly necessary if your affs are
    #  stored as uint8 or similar)
    # If you have many affinities of the exact same value the order they are processed
    # in may be fifo, so you can get annoying streaks.

    ### tmp comment out ###

    random_noise = np.random.randn(*affs.shape) * 0.001  # todo: parameterize?

    #######################

    # add smoothed affs, to solve a similar issue to the random noise. We want to bias
    # towards processing the central regions of objects first.

    ### tmp comment out ###

    smoothed_affs = (
        gaussian_filter(affs, sigma=sigma) - 0.5
    ) * 0.01  # todo: parameterize?

    #######################

    shift = np.array(
        [
            adjacent_edge_bias if max(offset) <= 1 else lr_edge_bias
            for offset in neighborhood
        ]
    ).reshape((-1, *((1,) * (len(affs.data.shape) - 1))))

    fragments_data = mws.agglom(
        affs + shift + random_noise + smoothed_affs,
        offsets=neighborhood,
        strides=strides,
    )

    return fragments_data


