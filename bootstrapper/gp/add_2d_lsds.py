import gunpowder as gp
import numpy as np

from lsd.train.gp import AddLocalShapeDescriptor
from lsd.train import LsdExtractor


class Add2DLSDs(AddLocalShapeDescriptor):
    """Create a 2D local segmentation shape discriptor to each voxel.

    Args:

        segmentation (:class:`ArrayKey`): The array storing the segmentation
            to use.

        descriptor (:class:`ArrayKey`): The array of the shape descriptor to
            generate.

        lsds_mask (:class:`ArrayKey`, optional): The array to store a binary mask
            the size of the descriptors. Background voxels, which do not have a
            descriptor, will be set to 0. This can be used as a loss scale
            during training, such that background is ignored.

        labels_mask (:class:`ArrayKey`, optional): The array to use as a mask
            for labels. Lsds connecting at least one masked out label will be
            masked out in lsds_mask.

        unlabelled (:class:`ArrayKey`, optional): A binary array to indicate
            unlabelled areas with 0. Lsds from labelled to unlabelled voxels are set
            to 0, lsds between unlabelled voxels are masked out (they will not be
            used for training).

        sigma (float or tuple of float): The context to consider to compute
            the shape descriptor in world units. This will be the standard
            deviation of a Gaussian kernel or the radius of the sphere.

        mode (string): Either ``gaussian`` or ``sphere``. Specifies how to
            accumulate local statistics: ``gaussian`` uses Gaussian convolution
            to compute a weighed average of statistics inside an object.
            ``sphere`` accumulates values in a sphere.

        components (string, optional): The components of the local shape descriptors to
            compute and return. Should be a string of integers chosen from 0 through
            6 (if 2D), in order. Example: "345". Defaults to all components.

            Component string lookup, where example component : "2D axes"

                mean offset (mean) : "01"
                orthogonal covariance (ortho) : "23"
                diagonal covariance (diag) : "4"
                size : "5"

            Example combinations:

                diag + size : "45"
                mean + diag + size : "0145"
                mean + ortho + diag : "01234"
                ortho + diag : "234"

        downsample (int, optional): Downsample the segmentation mask to extract
            the statistics with the given factore. Default is 1 (no
            downsampling).
    """

    def __init__(self, segmentation, descriptor, *args, **kwargs):

        super().__init__(segmentation, descriptor, *args, **kwargs)

        self.extractor = LsdExtractor(self.sigma[1:], self.mode, self.downsample)

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
        batch = gp.Batch()

        # create lsds mask array
        if self.lsds_mask and self.lsds_mask in request:

            if self.labels_mask:

                mask = self._create_mask(
                    old_batch, self.labels_mask, descriptor
                )  # , crop)

            else:

                mask = (labels != 0).astype(np.float32)

                mask_shape = len(mask.shape)

                assert mask.shape[-mask_shape:] == descriptor.shape[-mask_shape:]

                mask = np.array([mask] * descriptor.shape[0])

            if self.unlabelled:

                unlabelled_mask = self._create_mask(
                    old_batch, self.unlabelled, descriptor
                )

                mask = mask * unlabelled_mask

            batch[self.lsds_mask] = gp.Array(mask.astype(spec.dtype), spec.copy())

        batch[self.descriptor] = gp.Array(descriptor.astype(spec.dtype), spec)

        return batch

    def _create_mask(self, batch, mask, lsds):

        mask = batch.arrays[mask].data
        mask = np.array([mask] * lsds.shape[0])

        return mask
