import gunpowder as gp
import numpy as np
import random
from scipy.ndimage import gaussian_filter

from lsd.train.gp import AddLocalShapeDescriptor
from lsd.train import LsdExtractor


class SmoothAugment(gp.BatchFilter):
    def __init__(self, array, blur_range=(0.0, 1.0)):
        self.array = array
        self.range = blur_range

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

class Add2DLSDs(AddLocalShapeDescriptor):
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
        batch = gp.Batch()
        
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
                    old_batch, self.unlabelled, descriptor
                )

                mask = mask * unlabelled_mask

            batch[self.lsds_mask] = gp.Array(
                mask.astype(spec.dtype), spec.copy()
            )

        batch[self.descriptor] = gp.Array(descriptor.astype(spec.dtype), spec)

        return batch

    def _create_mask(self, batch, mask, lsds):

        mask = batch.arrays[mask].data
        mask = np.array([mask] * lsds.shape[0])

        return mask