import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def cc_blockwise(config):
    raise NotImplementedError("Blockwise connected components not implemented yet")


def cc_affs(config):
    import os
    import numpy as np
    from funlib.persistence import open_ds, prepare_ds
    from funlib.geometry import Roi
    from .cc import compute_connected_component_segmentation
    from scipy.ndimage import gaussian_filter
    from skimage.morphology import remove_small_objects

    affs_ds = config["affs_dataset"]
    frags_ds = config["fragments_dataset"]
    seg_container = config["seg_container"]
    seg_ds_prefix = config["seg_dataset_prefix"]
    mask_ds = config.get("mask_dataset", None)
    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)

    # required params
    threshold = config.get("threshold", 0.5)

    # optional params
    sigma = config.get("sigma", None)
    noise_eps = config.get("noise_eps", None)
    remove_debris = config.get("remove_debris", 0)

    # load affs
    affs = open_ds(affs_ds)

    # get total ROI
    if roi_offset is not None:
        roi = Roi(roi_offset, roi_shape)
    else:
        roi = affs.roi

    # load data
    affs_data = affs[roi][:3]

    # normalize
    if affs_data.dtype == np.uint8:
        affs_data = affs_data.astype(np.float32) / 255.0
    else:
        affs_data = affs_data.astype(np.float32)

    # load mask
    if mask_ds is not None:
        mask = open_ds(mask_ds)
        mask = mask[roi]
    else:
        mask = None

    if mask is not None:
        affs_data *= (mask > 0).astype(np.uint8) 

    # add shift and noise
    if sigma is not None or noise_eps is not None:
        shift = np.zeros_like(affs_data)
        if noise_eps is not None:
            shift += np.random.randn(*affs_data.shape) * noise_eps
        if sigma is not None:
            sigma = (0, *sigma)
            shift += gaussian_filter(affs_data, sigma=sigma) - affs_data
        affs_data += shift

    # threshold affs
    hard_affs = affs_data > threshold

    # compute connected components
    fragments_data = compute_connected_component_segmentation(hard_affs)

    # write fragments
    frags = prepare_ds(
        frags_ds,
        shape=fragments_data.shape,
        offset=roi.offset,
        voxel_size=affs.voxel_size,
        axis_names=affs.axis_names[1:],
        dtype=np.uint64,
        units=affs.units,
    )
    frags[roi] = fragments_data

    # remove small debris
    if remove_debris > 0:
        fragments_dtype = fragments_data.dtype
        fragments_data = fragments_data.astype(np.int64)
        fragments_data = remove_small_objects(
            fragments_data, min_size=remove_debris
        )
        fragments_data = fragments_data.astype(fragments_dtype)

    # write segmentation
    seg_ds_name = os.path.join(seg_container, seg_ds_prefix, "cc", str(threshold))
    seg = prepare_ds(
        seg_ds_name,
        shape=fragments_data.shape,
        offset=roi.offset,
        voxel_size=affs.voxel_size,
        axis_names=affs.axis_names[1:],
        dtype=np.uint64,
        units=affs.units,
    )
    seg[roi] = fragments_data

def cc_segmentation(config):
    # blockwise or not
    blockwise = config.get("blockwise", False)

    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)

    if roi_offset is not None:
        config['roi_offset'] = list(map(int, roi_offset.strip().split(" ")))
        config['roi_shape'] = list(map(int, roi_shape.strip().split(" ")))

    if blockwise:
        cc_blockwise(config)
    else:
        cc_affs(config)