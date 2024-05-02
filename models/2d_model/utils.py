import logging
from gunpowder import BatchFilter
from gunpowder.array import Array
from gunpowder.batch_request import BatchRequest
from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
import random

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, maximum_filter, generate_binary_structure
from skimage.measure import label
from skimage.morphology import disk, star, ellipse
from skimage.segmentation import expand_labels, watershed

from gunpowder.nodes.add_affinities import seg_to_affgraph
from lsd.train.gp import AddLocalShapeDescriptor
from lsd.train import LsdExtractor

logger = logging.getLogger(__name__)

class SliceArray(BatchFilter):
    def __init__(self, array, slice_obj):
        self.array = array
        self.slice_obj = slice_obj

    def process(self, batch, request):

        array = batch[self.array].data
        batch[self.array].data = array[self.slice_obj]


class SmoothArray(BatchFilter):
    def __init__(self, array, blur_range=None):
        self.array = array
        self.range = blur_range

        if self.range is None:
            self.range = [0.0,1.0]

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


class CustomAffs(BatchFilter):
    """Add an array with affinities for a given label array and neighborhood to
    the batch. Affinity values are created one for each voxel and entry in the
    neighborhood list, i.e., for each voxel and each neighbor of this voxel.
    Values are 1 iff both labels (of the voxel and the neighbor) are equal and
    non-zero.

    Args:

        affinity_neighborhood (``list`` of array-like):

            List of offsets for the affinities to consider for each voxel.

        labels (:class:`ArrayKey`):

            The array to read the labels from.

        affinities (:class:`ArrayKey`):

            The array to generate containing the affinities.

        labels_mask (:class:`ArrayKey`, optional):

            The array to use as a mask for ``labels``. Affinities connecting at
            least one masked out label will be masked out in
            ``affinities_mask``. If not given, ``affinities_mask`` will contain
            ones everywhere (if requested).

        unlabelled (:class:`ArrayKey`, optional):

            A binary array to indicate unlabelled areas with 0. Affinities from
            labelled to unlabelled voxels are set to 0, affinities between
            unlabelled voxels are masked out (they will not be used for
            training).

        affinities_mask (:class:`ArrayKey`, optional):

            The array to generate containing the affinitiy mask, as derived
            from parameter ``labels_mask``.
    """

    def __init__(
        self,
        affinity_neighborhood,
        labels,
        affinities,
        labels_mask=None,
        unlabelled=None,
        affinities_mask=None,
        dtype=np.uint8,
    ):
        self.affinity_neighborhood = np.array(affinity_neighborhood)
        self.labels = labels
        self.unlabelled = unlabelled
        self.labels_mask = labels_mask
        self.affinities = affinities
        self.affinities_mask = affinities_mask
        self.dtype = dtype

    def setup(self):
        assert self.labels in self.spec, (
            "Upstream does not provide %s needed by " "AddAffinities" % self.labels
        )

        voxel_size = self.spec[self.labels].voxel_size

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

        spec = self.spec[self.labels].copy()
        if spec.roi is not None:
            spec.roi = spec.roi.grow(self.padding_neg, -self.padding_pos)
        spec.dtype = self.dtype

        self.provides(self.affinities, spec)
        if self.affinities_mask:
            self.provides(self.affinities_mask, spec)
        self.enable_autoskip()

    def prepare(self, request):
        deps = BatchRequest()

        # grow labels ROI to accomodate padding
        labels_roi = request[self.affinities].roi.grow(
            -self.padding_neg, self.padding_pos
        )
        deps[self.labels] = request[self.affinities].copy()
        deps[self.labels].dtype = None
        deps[self.labels].roi = labels_roi

        if self.labels_mask:
            deps[self.labels_mask] = deps[self.labels].copy()
        if self.unlabelled:
            deps[self.unlabelled] = deps[self.labels].copy()

        return deps

    def process(self, batch, request):
        outputs = Batch()

        affinities_roi = request[self.affinities].roi

        logger.debug("computing ground-truth affinities from labels")

        affinities = seg_to_affgraph(
            batch.arrays[self.labels].data.astype(np.int32), self.affinity_neighborhood
        ).astype(self.dtype)

        # crop affinities to requested ROI
        offset = affinities_roi.offset
        shift = -offset - self.padding_neg
        crop_roi = affinities_roi.shift(shift)
        crop_roi /= self.spec[self.labels].voxel_size
        crop = crop_roi.get_bounding_box()

        logger.debug("cropping with " + str(crop))
        affinities = affinities[(slice(None),) + crop]
        
        # remove z-channel of affinities
        affinities = affinities[1:3]

        spec = self.spec[self.affinities].copy()
        spec.roi = affinities_roi
        outputs.arrays[self.affinities] = Array(affinities, spec)

        if self.affinities_mask and self.affinities_mask in request:
            if self.labels_mask:
                logger.debug(
                    "computing ground-truth affinities mask from " "labels mask"
                )
                affinities_mask = seg_to_affgraph(
                    batch.arrays[self.labels_mask].data.astype(np.int32),
                    self.affinity_neighborhood,
                )
                affinities_mask = affinities_mask[(slice(None),) + crop]

            else:
                affinities_mask = np.ones_like(affinities)

            if self.unlabelled:
                # 1 for all affinities between unlabelled voxels
                unlabelled = 1 - batch.arrays[self.unlabelled].data
                unlabelled_mask = seg_to_affgraph(
                    unlabelled.astype(np.int32), self.affinity_neighborhood
                )
                unlabelled_mask = unlabelled_mask[(slice(None),) + crop]

                # 0 for all affinities between unlabelled voxels
                unlabelled_mask = 1 - unlabelled_mask

                # combine with mask
                affinities_mask = affinities_mask * unlabelled_mask[1:3]

            affinities_mask = affinities_mask.astype(affinities.dtype)
            outputs.arrays[self.affinities_mask] = Array(affinities_mask, spec)

        else:
            if self.labels_mask is not None:
                logger.warning(
                    "GT labels does have a mask, but affinities "
                    "mask is not requested."
                )

        # Should probably have a better way of handling arbitrary batch attributes
        batch.affinity_neighborhood = self.affinity_neighborhood[1:]

        return outputs





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
            labels_sec = labels[z]

            descriptor_sec = self.extractor.get_descriptors(
                segmentation=labels_sec, voxel_size=spec.voxel_size[1:]
            )

            descriptor[:, z] = descriptor_sec

        old_batch = batch
        batch = Batch()
        
        # create lsds mask array
        if self.lsds_mask and self.lsds_mask in request:

            if self.labels_mask:

                mask = self._create_mask(old_batch, self.labels_mask, descriptor)#, crop)

            else:

                mask = (labels != 0).astype(
                    np.float32
                )

                mask_shape = len(mask.shape)

                assert mask.shape[-mask_shape:] == descriptor.shape[-mask_shape:]

                mask = np.array([mask] * descriptor.shape[0])

            if self.unlabelled:

                unlabelled_mask = self._create_mask(
                    old_batch, self.unlabelled, descriptor#, crop
                )

                mask = mask * unlabelled_mask

            batch[self.lsds_mask] = Array(
                mask.astype(spec.dtype), spec.copy()
            )

        batch[self.descriptor] = Array(descriptor.astype(spec.dtype), spec)

        return batch


    def _create_mask(self, batch, mask, lsds):#, #crop):

        mask = batch.arrays[mask].data

        mask = np.array([mask] * lsds.shape[0])

        #mask = mask[(slice(None),) + crop]

        return mask
