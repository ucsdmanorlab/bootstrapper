from lsd.train import LsdExtractor
from gunpowder import BatchFilter, Array, BatchRequest, Batch
import logging
import numpy as np

from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import ball, disk

logger = logging.getLogger(__name__)


class AddLSDErrors(BatchFilter):

    """Compute a local shape descriptor error map and error mask from segmentation 
    and predicted descriptors.

    Args:

        segmentation (:class:`ArrayKey`): The array storing the segmentation
            to use.

        seg_descriptor (:class:`ArrayKey`): The array of the shape descriptor to
            generate.
        
        pred_descriptor (:class:`ArrayKey`): The array of the shape descriptor to
            compute errors against.

        error_map (:class:`ArrayKey`): The array storing the voxel-wise difference
            between seg_descriptor and pred_descriptor. 
        
        error_mask (:class:`ArrayKey`): The array storing the thresholded and closed
            error map. Closed refers to binary closing. 

        thresholds (tuple of float): Floor and ceiling values to use to threshold
            the error map.

        labels_mask (:class:`ArrayKey`, optional): The array to use as a mask
            for the errors. Errors are zero in masked out regions. 

        sigma (float or tuple of float): The context to consider to compute
            the shape descriptor in world units. This will be the standard
            deviation of a Gaussian kernel or the radius of the sphere.

        mode (string): Either ``gaussian`` or ``sphere``. Specifies how to
            accumulate local statistics: ``gaussian`` uses Gaussian convolution
            to compute a weighed average of statistics inside an object.
            ``sphere`` accumulates values in a sphere.

        components (string, optional): The components of the local shape descriptors to
            compute and return. Should be a string of integers chosen from 0 through 9 (if 3D)
            or 6 (if 2D), in order. Example: "0129" or "345". Defaults to all components.

            Component string lookup, where example component : "3D axes", "2D axes"

                mean offset (mean) : "012", "01"
                orthogonal covariance (ortho) : "345", "23"
                diagonal covariance (diag) : "678", "4"
                size : "9", "5"

            Example combinations:

                diag + size : "6789", "45"
                mean + diag + size : "0126789", "0145"
                mean + ortho + diag : "012345678", "01234"
                ortho + diag : "345678", "234"

        downsample (int, optional): Downsample the segmentation mask to extract
            the statistics with the given factore. Default is 1 (no
            downsampling).
    """

    def __init__(
        self,
        segmentation,
        seg_descriptor,
        pred_descriptor,
        error_map,
        error_mask,
        thresholds=(0.1,1.0),
        labels_mask=None,
        sigma=5.0,
        mode="gaussian",
        components=None,
        downsample=1,
        array_specs=None,
    ):

        self.segmentation = segmentation
        self.seg_descriptor = seg_descriptor
        self.pred_descriptor = pred_descriptor
        self.error_map = error_map
        self.error_mask = error_mask
        self.thresholds = thresholds
        self.labels_mask = labels_mask
        self.components = components
        self.array_specs = {} if array_specs is None else array_specs 

        try:
            self.sigma = tuple(sigma)
        except:
            self.sigma = (sigma,) * 3

        self.mode = mode
        self.downsample = downsample
        self.voxel_size = None
        self.context = None
        self.skip = False

        self.extractor = LsdExtractor(self.sigma, self.mode, self.downsample)

    def setup(self):

        spec = self.spec[self.segmentation].copy()
        spec.dtype = np.float32

        mask_spec = spec.copy()
        mask_spec.dtype = np.uint8

        self.voxel_size = spec.voxel_size
        self.provides(self.seg_descriptor, spec)

        if self.error_map in self.array_specs:
            self.provides(self.error_map, self.array_specs[self.error_map].copy())
        else:
            self.provides(self.error_map, spec)

        if self.error_mask in self.array_specs:
            self.provides(self.error_mask, self.array_specs[self.error_mask].copy())
        else:
            self.provides(self.error_mask, mask_spec)

        if self.mode == "gaussian":
            self.context = tuple(s * 3 for s in self.sigma)
        elif self.mode == "sphere":
            self.context = tuple(self.sigma)
        else:
            raise RuntimeError("Unkown mode %s" % self.mode)

    def prepare(self, request):
        deps = BatchRequest()
        if self.seg_descriptor in request:

            dims = len(request[self.seg_descriptor].roi.get_shape())

            if dims == 2:
                self.context = self.context[0:2]

            # increase segmentation ROI to fit Gaussian
            context_roi = request[self.seg_descriptor].roi.grow(self.context, self.context)

            # ensure context roi is multiple of voxel size
            context_roi = context_roi.snap_to_grid(self.voxel_size, mode="shrink")

            grown_roi = request[self.segmentation].roi.union(context_roi)

            deps[self.segmentation] = request[self.seg_descriptor].copy()
            deps[self.segmentation].roi = grown_roi
            
            deps[self.pred_descriptor] = request[self.pred_descriptor].copy()
            deps[self.pred_descriptor].roi = grown_roi

        else:
            self.skip = True

        if self.labels_mask:
            deps[self.labels_mask] = deps[self.segmentation].copy()

        return deps

    def process(self, batch, request):
        if self.skip:
            return

        dims = len(self.voxel_size)

        segmentation_array = batch[self.segmentation]

        # get voxel roi of requested descriptors
        # this is the only region in
        # which we have to compute the descriptors
        seg_roi = segmentation_array.spec.roi
        seg_descriptor_roi = request[self.seg_descriptor].roi
        voxel_roi_in_seg = (
            seg_roi.intersect(seg_descriptor_roi) - seg_roi.get_offset()
        ) / self.voxel_size

        crop = voxel_roi_in_seg.get_bounding_box()

        seg_descriptor = self.extractor.get_descriptors(
            segmentation=segmentation_array.data,
            components=self.components,
            voxel_size=self.voxel_size,
            roi=voxel_roi_in_seg,
        )

        # create seg_descriptor array
        seg_descriptor_spec = self.spec[self.seg_descriptor].copy()
        seg_descriptor_spec.roi = request[self.seg_descriptor].roi.copy()
        seg_descriptor_array = Array(seg_descriptor, seg_descriptor_spec)

        # load pred_descriptor array and labels_mask array
        pred_descriptor_array = batch[self.pred_descriptor].crop(seg_descriptor_spec.roi)
        if self.labels_mask:
            labels_mask_array = batch[self.labels_mask].crop(seg_descriptor_spec.roi)

        # create error map array
        error_map_array = self._create_diff(
                seg_descriptor_array.data,
                pred_descriptor_array.data,
                None if not self.labels_mask else labels_mask_array.data)

        # create error mask array
        error_mask_array = self._create_mask(
                error_map_array,
                self.thresholds)
        mask_spec = seg_descriptor_spec.copy()
        mask_spec.dtype = np.uint8

        # Create new batch for descriptor:
        batch = Batch()
        batch[self.seg_descriptor] = seg_descriptor_array
        batch[self.error_map] = Array(error_map_array, seg_descriptor_spec)
        batch[self.error_mask] = Array(error_mask_array, mask_spec)

        return batch

    def _create_diff(self, a_data, b_data, mask_data=None):

        diff_data = np.sum((a_data - b_data)**2, axis=0)
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

        #TODO: make erode-dilate optional
        # dilate/erode
        z_struct = np.stack([ball(1)[0],]*3)
        xy_struct = np.stack([np.zeros((3,3)),disk(1),np.zeros((3,3))])

        # to remove minor pixel-wise differences along xy boundaries
        o_data = binary_erosion(o_data, xy_struct, iterations=4)
        o_data = binary_dilation(o_data, xy_struct, iterations=4)

        # to join gaps between z-splits in error mask
        o_data = binary_dilation(o_data, z_struct)
        o_data = binary_erosion(o_data, z_struct)

        o_data = o_data.astype(np.uint8)
        return o_data