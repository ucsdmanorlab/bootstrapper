import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def waterz_pipeline(config):
    from .blockwise.hglom.frags import extract_fragments
    from .blockwise.hglom.agglom import agglomerate
    from .blockwise.hglom.luts import find_segments
    from .blockwise.hglom.extract import extract_segmentations

    extract_fragments(config)
    agglomerate(config)
    find_segments(config)
    extract_segmentations(config)


def simple_watershed(config):
    import os
    import numpy as np
    from funlib.persistence import open_ds, prepare_ds
    from funlib.geometry import Roi
    from scipy.ndimage import gaussian_filter
    from .ws import watershed_from_affinities
    import waterz

    affs_ds = config["affs_dataset"]
    frags_ds = config["fragments_dataset"]
    seg_container = config["seg_container"]
    seg_ds_prefix = config["seg_dataset_prefix"]
    mask_ds = config.get("mask_dataset", None)
    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)

    # optional waterz params
    thresholds = config.get("thresholds", [0.5])
    fragments_in_xy = config.get("fragments_in_xy", True)
    min_seed_distance = config.get("min_seed_distance", 10)
    merge_function = config.get("merge_function", "mean")
    sigma = config.get("sigma", None)
    noise_eps = config.get("noise_eps", None)
    bias = config.get("bias", None)

    assert len(thresholds) == 1, "Only one threshold supported for simple watershed"

    waterz_merge_function = {
        "hist_quant_10": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>",
        "hist_quant_10_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>",
        "hist_quant_25": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
        "hist_quant_25_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>",
        "hist_quant_50": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>",
        "hist_quant_50_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>",
        "hist_quant_75": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
        "hist_quant_75_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>",
        "hist_quant_90": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>",
        "hist_quant_90_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>",
        "mean": "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    }[merge_function]

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

    # shift affs with noise, smoothing, and bias
    if sigma is not None or noise_eps is not None or bias is not None:
        shift = np.zeros_like(affs_data)

        if noise_eps is not None:
            shift += np.random.randn(*affs_data.shape) * noise_eps

        if sigma is not None:
            sigma = (0, *sigma)
            shift += gaussian_filter(affs_data, sigma=sigma) - affs_data

        if bias is not None:
            if type(bias) == float:
                bias = [bias] * affs_data.shape[0]
            else:
                assert len(bias) == affs_data.shape[0]
            
            shift += np.array([bias]).reshape(
                (-1, *((1,) * (len(affs.shape) - 1)))
            )

        affs_data += shift

    # watershed
    fragments_data, n = watershed_from_affinities(
        affs_data,
        fragments_in_xy=fragments_in_xy,
        return_seeds=False,
        min_seed_distance=min_seed_distance,
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

    # agglomerate
    generator = waterz.agglomerate(
        affs_data,
        thresholds=thresholds,
        fragments=fragments_data.copy(),
        scoring_function=waterz_merge_function
    )

    segmentation = next(generator)

    # write segmentation
    seg_ds_name = os.path.join(seg_container, seg_ds_prefix, "watershed", str(thresholds[0]))
    seg = prepare_ds(
        seg_ds_name,
        shape=segmentation.shape,
        offset=roi.offset,
        voxel_size=affs.voxel_size,
        axis_names=affs.axis_names[1:],
        dtype=np.uint64,
        units=affs.units,
    )
    seg[roi] = segmentation


def watershed_segmentation(config):
    # blockwise or not
    blockwise = config.get("blockwise", False)

    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)
    block_shape = config.get("block_shape", None)

    if roi_offset is not None:
        config['roi_offset'] = list(map(int, roi_offset.strip().split(" ")))
        config['roi_shape'] = list(map(int, roi_shape.strip().split(" ")))

    if blockwise:
        if block_shape == "roi":
            config['blockwise'] = False
        waterz_pipeline(config)
    else:
        simple_watershed(config)
