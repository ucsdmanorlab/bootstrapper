import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def volara_pipeline(config):
    raise NotImplementedError("Blockwise mutex segmentation coming soon!")


def simple_mutex(config):
    import os
    import numpy as np
    from funlib.persistence import open_ds, prepare_ds
    from funlib.geometry import Roi
    from .mws import mwatershed_from_affinities
    from skimage.morphology import remove_small_objects

    affs_ds = config["affs_dataset"]
    frags_ds = config["fragments_dataset"]
    seg_ds_prefix = config["seg_dataset_prefix"]
    mask_ds = config.get("mask_dataset", None)
    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)

    # required mws params
    neighborhood = config.get("aff_neighborhood", None)
    bias = config.get("bias", None)

    # optional mws params
    sigma = config.get("sigma", None)
    noise_eps = config.get("noise_eps", None)
    strides = config.get("strides", None)
    randomized_strides = config.get("randomized_strides", False)
    remove_debris = config.get("remove_debris", 0)

    # load affs
    affs = open_ds(affs_ds)

    # validate neighborhood and bias
    if neighborhood is None:
        raise ValueError("Affinities neighborrhood must be provided")
    if bias is None:
        raise ValueError("Affinities bias must be provided")

    assert (
        len(neighborhood) == affs.shape[0]
    ), "Number of offsets must match number of affinities channels"
    assert len(neighborhood) == len(
        bias
    ), "Numbes of biases must match number of affinities channels"

    # get total ROI
    if roi_offset is not None:
        roi = Roi(roi_offset, roi_shape)
    else:
        roi = affs.roi

    # load data
    affs_data = affs[roi]

    # normalize
    if affs_data.dtype == np.uint8:
        affs_data = affs_data.astype(np.float64) / 255.0
    else:
        affs_data = affs_data.astype(np.float64)

    # load mask
    if mask_ds is not None:
        mask = open_ds(mask_ds)
        mask = mask[roi]
    else:
        mask = None

    if mask is not None:
        affs_data *= (mask > 0).astype(np.uint8)

    # watershed
    fragments_data = mwatershed_from_affinities(
        affs_data, neighborhood, bias, sigma, noise_eps, strides, randomized_strides
    )

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
        fragments_data = remove_small_objects(fragments_data, min_size=remove_debris)
        fragments_data = fragments_data.astype(fragments_dtype)

    # write segmentation
    bias_str = "--".join(
        "_".join([str(x) for x in bias]), 
        "_".join([str(x) for x in sigma]),
        str(noise_eps),
    seg_ds_name = os.path.join(seg_ds_prefix, bias_str)
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


def mutex_watershed_segmentation(config):
    # blockwise or not
    blockwise = config.get("blockwise", False)

    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)
    block_shape = config.get("block_shape", None)

    if roi_offset is not None:
        config["roi_offset"] = list(map(int, roi_offset.strip().split(" ")))
        config["roi_shape"] = list(map(int, roi_shape.strip().split(" ")))

    if blockwise:
        if block_shape == "roi":
            config["blockwise"] = False
        volara_pipeline(config)
    else:
        simple_mutex(config)
