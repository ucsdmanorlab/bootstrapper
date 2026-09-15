import numpy as np
from scipy.ndimage import mean as ndi_mean

from volara.blockwise import ExtractFrags as VolaraExtractFrags
from volara.tmp import replace_values


class ExtractFrags(VolaraExtractFrags):
    """ExtractFrags with the fragment filter written back into the array.

    ``replace_values`` returns a new array, so the parent's filter result is lost.
    """

    def filter_avg_fragments(self, affs, fragments_data, filter_value):
        average_affs = np.mean(affs[0:3], axis=0)
        fragment_ids = np.unique(fragments_data)
        means = ndi_mean(average_affs, fragments_data, fragment_ids)
        filtered = np.array(
            [f for f, m in zip(fragment_ids, means) if m < filter_value],
            dtype=fragments_data.dtype,
        )
        fragments_data[:] = replace_values(
            fragments_data, filtered, np.zeros_like(filtered)
        )
