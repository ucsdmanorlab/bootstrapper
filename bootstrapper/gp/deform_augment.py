import gunpowder as gp
import numpy as np
from scipy.ndimage import map_coordinates


class DeformAugment(gp.DeformAugment):
    """gp.DeformAugment with the transform sampled at the true control-point positions.

    gunpowder 1.4.4 and 1.5.0 resample the control grid with an endpoint-aligned zoom per
    requested ROI, so raw and labels get different zooms and do not line up.
    """

    def _DeformAugment__sample_transform(self, transformation, output_spec):
        roi, voxel_size = output_spec.roi, output_spec.voxel_size
        begin, step = transformation.spec.roi.begin, transformation.spec.voxel_size
        axes = [
            (
                roi.begin[d]
                + np.arange(roi.shape[d] // voxel_size[d]) * voxel_size[d]
                - begin[d]
            )
            / step[d]
            for d in range(roi.dims)
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        data = np.stack(
            [
                map_coordinates(transformation.data[d], grid, order=1, mode="nearest")
                for d in range(roi.dims)
            ]
        )
        return gp.Array(
            data.astype(np.float32), gp.ArraySpec(roi, voxel_size, interpolatable=True)
        )
