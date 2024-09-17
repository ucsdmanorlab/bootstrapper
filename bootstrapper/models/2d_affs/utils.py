import gunpowder as gp
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from skimage.measure import label


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
