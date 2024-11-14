from gunpowder import BatchFilter, Array, BatchRequest, Batch, Coordinate
import logging
import numpy as np

from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import ball, disk

from gunpowder.nodes.add_affinities import seg_to_affgraph

logger = logging.getLogger(__name__)


class AddAffErrors(BatchFilter):
    """
    Compute an affinity error map and error mask from segmentation
    and predicted affinities.
    """

    def __init__(
        self,
        segmentation,
        seg_affs,
        pred_affs,
        error_map,
        error_mask,
        neighborhood,
        labels_mask=None,
        thresholds=(0.1, 1.0),
        array_specs=None,
    ):
        self.segmentation = segmentation
        self.affinity_neighborhood = np.array(neighborhood)
        self.seg_affs = seg_affs
        self.pred_affs = pred_affs
        self.error_map = error_map
        self.error_mask = error_mask
        self.thresholds = thresholds
        self.labels_mask = labels_mask
        self.array_specs = {} if array_specs is None else array_specs

        # get max offset in each dimension from neighborhood
        self.context = Coordinate(
            [
                max(abs(offset[dim]) for offset in self.affinity_neighborhood)
                for dim in range(3)
            ]
        )

    # def setup(self):
    #     spec = self.spec[self.segmentation].copy()
    #     spec.dtype = np.float32

    #     mask_spec = spec.copy()
    #     mask_spec.dtype = np.uint8

    #     self.voxel_size = spec.voxel_size
    #     self.provides(self.seg_affs, spec)

    #     if self.error_map in self.array_specs:
    #         self.provides(self.error_map, self.array_specs[self.error_map].copy())
    #     else:
    #         self.provides(self.error_map, spec)

    #     if self.error_mask in self.array_specs:
    #         self.provides(self.error_mask, self.array_specs[self.error_mask].copy())
    #     else:
    #         self.provides(self.error_mask, mask_spec)

    def setup(self):

        voxel_size = self.spec[self.segmentation].voxel_size

        dims = self.affinity_neighborhood.shape[1]
        self.padding_neg = (
            Coordinate(
                min([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
            )
            * voxel_size
        )

        self.padding_pos = (
            Coordinate(
                max([0] + [a[d] for a in self.affinity_neighborhood])
                for d in range(dims)
            )
            * voxel_size
        )

        logger.debug("padding neg: " + str(self.padding_neg))
        logger.debug("padding pos: " + str(self.padding_pos))

        spec = self.spec[self.segmentation].copy()
        if spec.roi is not None:
            spec.roi = spec.roi.grow(self.padding_neg, -self.padding_pos)
        spec.dtype = np.float32

        self.provides(self.seg_affs, spec)

        if self.error_map in self.array_specs:
            self.provides(self.error_map, self.array_specs[self.error_map].copy())
        else:
            self.provides(self.error_map, spec)

        if self.error_mask in self.array_specs:
            self.provides(self.error_mask, self.array_specs[self.error_mask].copy())
        else:
            self.provides(self.error_mask, spec)

    def prepare(self, request):
        deps = BatchRequest()

        # increase segmentation ROI to fit neighborhood
        grown_roi = request[self.seg_affs].roi.grow(-self.padding_neg, self.padding_pos)
        deps[self.segmentation] = request[self.seg_affs].copy()
        deps[self.segmentation].dtype = None
        deps[self.segmentation].roi = grown_roi

        deps[self.pred_affs] = request[self.seg_affs].copy()
        deps[self.pred_affs].dtype = None  # np.float32
        deps[self.pred_affs].roi = grown_roi

        if self.labels_mask:
            deps[self.labels_mask] = deps[self.segmentation].copy()

        return deps

    def process(self, batch, request):

        seg_array = batch[self.segmentation]
        seg_affs = seg_to_affgraph(seg_array.data, self.affinity_neighborhood).astype(
            np.float32
        )

        # crop affinities to requested ROI
        seg_affs_spec = self.spec[self.seg_affs].copy()
        seg_affs_spec.roi = batch[self.segmentation].spec.roi.copy()
        seg_affs_array = Array(seg_affs, seg_affs_spec)
        seg_affs_array = seg_affs_array.crop(request[self.seg_affs].roi)

        pred_affs_array = batch[self.pred_affs].crop(request[self.seg_affs].roi)

        if self.labels_mask:
            mask_array = batch[self.labels_mask].crop(request[self.seg_affs].roi)
        else:
            mask_array = None

        error_map_array = self._create_diff(
            seg_affs_array.data,
            pred_affs_array.data,
            mask_array.data if mask_array is not None else None,
        )
        error_mask_array = self._create_mask(error_map_array, self.thresholds)

        mask_spec = seg_affs_array.spec.copy()
        mask_spec.dtype = np.uint8

        batch = Batch()
        batch[self.seg_affs] = seg_affs_array
        batch[self.error_map] = Array(error_map_array, seg_affs_array.spec)
        batch[self.error_mask] = Array(error_mask_array, mask_spec)

        return batch

    def _create_diff(self, a_data, b_data, mask_data=None):

        diff_data = np.sum((a_data - b_data) ** 2, axis=0)
        if mask_data is not None:
            diff_data *= mask_data

        # normalize
        max_value = np.max(diff_data)

        if max_value > 0:
            diff_data /= max_value
        else:
            diff_data[:] = 0

        return diff_data

    def _create_mask(self, i_data, thresholds):

        floor, ceil = thresholds

        # threshold
        o_data = (i_data > floor) & (i_data < ceil)

        # # TODO: make erode-dilate optional
        # # dilate/erode
        # z_struct = np.stack(
        #     [
        #         ball(1)[0],
        #     ]
        #     * 3
        # )
        # xy_struct = np.stack([np.zeros((3, 3)), disk(1), np.zeros((3, 3))])

        # # to remove minor pixel-wise differences along xy boundaries
        # o_data = binary_erosion(o_data, xy_struct, iterations=4)
        # o_data = binary_dilation(o_data, xy_struct, iterations=4)

        # # to join gaps between z-splits in error mask
        # o_data = binary_dilation(o_data, z_struct)
        # o_data = binary_erosion(o_data, z_struct)

        o_data = o_data.astype(np.uint8)
        return o_data
