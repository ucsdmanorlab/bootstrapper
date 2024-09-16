import gunpowder as gp
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from skimage.measure import label

from lsd.train.gp import AddLocalShapeDescriptor
from lsd.train import LsdExtractor


def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):
    method_padding = gp.Coordinate((0, sigma * 3, sigma * 3,))

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (gp.Coordinate([i / 2 for i in [0, diag, diag]]) + method_padding),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


class CreateMask(gp.BatchFilter):
  def __init__(self, in_array, out_array):
    self.in_array = in_array
    self.out_array = out_array

  def setup(self):
    # tell downstream nodes about the new array

    spec = self.spec[self.in_array].copy()
    spec.dtype = np.uint8

    self.provides(
      self.out_array,
      spec)

  def prepare(self, request):
    # to deliver mask, we need labels data in the same ROI
    deps = gp.BatchRequest()

    request_spec = request[self.out_array].copy()
    request_spec.dtype = request[self.in_array].dtype

    deps[self.in_array] = request_spec
 
    return deps

  def process(self, batch, request):
    # get the data from in_array and mask it
    data = batch[self.in_array].data

    # mask and convert to uint8
    data = (data > 0).astype(np.uint8)

    # create the array spec for the new array
    spec = batch[self.in_array].spec.copy()
    spec.roi = request[self.out_array].roi.copy()
    spec.dtype = np.uint8

    # create a new batch to hold the new array
    batch = gp.Batch()

    # create a new array
    masked = gp.Array(data, spec)

    # store it in the batch
    batch[self.out_array] = masked

    # return the new batch
    return batch


class Renumber(gp.BatchFilter):
    def __init__(self, labels):
        self.labels = labels

    def process(self, batch, request):
        components = batch.arrays[self.labels].data
        dtype = components.dtype

        components = label(components, connectivity=1)
        batch.arrays[self.labels].data = components.astype(dtype)


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

                array[z] = np.array(gaussian_filter(array_sec, sigma=sigma)).astype(
                    array_sec.dtype
                )

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
            array = np.array(gaussian_filter(array, sigma=sigma)).astype(array.dtype)

        else:
            raise AssertionError("array shape is not 2d, 3d, or multi-channel 3d")

        batch[self.array].data = array


class Add2DLSDs(AddLocalShapeDescriptor):
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
