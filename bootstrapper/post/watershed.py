import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _ws_shift_name(noise_eps, sigma, bias, min_seed_distance):
    parts = []
    if noise_eps is not None:
        parts.append(f"eps{noise_eps}")
    if sigma is not None:
        parts.append(f"sigma{'_'.join(map(str, sigma))}")
    if bias is not None:
        b = bias if isinstance(bias, (list, tuple)) else [bias]
        parts.append(f"bias{'_'.join(map(str, b))}")
    prefix = "--".join(parts)
    prefix = f"{prefix}--" if prefix else ""
    return f"{prefix}minseed{min_seed_distance}"


def waterz_pipeline(config):
    import os
    from pathlib import Path

    import numpy as np
    from funlib.geometry import Coordinate, Roi
    from funlib.persistence import open_ds
    from funlib.segment.graphs.impl import connected_components
    from volara.blockwise import Relabel
    from volara.datasets import Labels, Raw
    from volara.dbs import SQLite, PostgreSQL
    from volara.lut import LUT

    from .blockwise.watershed_frags import WatershedFrags
    from .blockwise.waterz_agglom import WaterzAgglom, WATERZ_MERGE_FUNCTIONS
    from ..blockwise import run_volara_task

    affs_dataset = config["affs_dataset"]
    fragments_dataset_prefix = config["fragments_dataset"]
    seg_dataset_prefix = config["seg_dataset_prefix"]
    lut_dir = config["lut_dir"]
    db_config = config["db"]
    mask_dataset = config.get("mask_dataset")

    # watershed fragment params (same as simple_watershed, plus seed_eps /
    # epsilon_agglomerate for blockwise)
    fragments_in_xy = config.get("fragments_in_xy", True)
    min_seed_distance = config.get("min_seed_distance", 10)
    seed_eps = config.get("seed_eps")
    epsilon_agglomerate = config.get("epsilon_agglomerate", 0.0)
    sigma = config.get("sigma")
    noise_eps = config.get("noise_eps")
    bias = config.get("bias")
    filter_fragments = config.get("filter_fragments", 0.0)
    remove_debris = config.get("remove_debris", 0)

    # waterz agglomeration params
    thresholds = config.get("thresholds", [0.2, 0.35, 0.5])
    merge_function = config.get("merge_function", "mean")
    waterz_merge_function = WATERZ_MERGE_FUNCTIONS[merge_function]

    # blockwise params
    roi_offset = config.get("roi_offset")
    roi_shape = config.get("roi_shape")
    blockwise = config.get("blockwise", False)
    num_workers = config.get("num_workers", 1) if blockwise else 1
    block_shape = config.get("block_shape")
    context = config.get("context")

    affs = open_ds(affs_dataset)

    if roi_offset is not None:
        total_roi = Roi(roi_offset, roi_shape)
    else:
        total_roi = affs.roi
    roi = (total_roi.offset, total_roi.shape)

    if blockwise:
        block_size = (
            Coordinate(block_shape) if block_shape else Coordinate(affs.chunk_shape[1:])
        )
        ctx = (
            Coordinate(context)
            if context
            else Coordinate([max(1, s // 8) for s in block_size])
        )
    else:
        block_size = Coordinate(affs.shape[1:])
        ctx = Coordinate([0] * affs.roi.dims)

    shift_name = _ws_shift_name(noise_eps, sigma, bias, min_seed_distance)
    frags_ds_name = str(Path(fragments_dataset_prefix) / shift_name)

    affinities = Raw(store=affs_dataset)
    mask_data = Raw(store=mask_dataset) if mask_dataset else None
    if "db_file" in db_config:
        db = SQLite(path=db_config["db_file"], edge_attrs={"merge_score": "float"})
    else:
        db = PostgreSQL(
            name=db_config["db_name"],
            host=db_config["db_host"],
            user=db_config["db_user"],
            password=db_config["db_password"],
            edge_attrs={"merge_score": "float"},
        )
    fragments = Labels(store=frags_ds_name)
    os.makedirs(lut_dir, exist_ok=True)

    # fragments via seeded watershed
    frags_task = WatershedFrags(
        db=db,
        affs_data=affinities,
        frags_data=fragments,
        mask_data=mask_data,
        block_size=block_size,
        context=ctx,
        num_workers=num_workers,
        roi=roi,
        fragments_in_xy=fragments_in_xy,
        min_seed_distance=min_seed_distance,
        seed_eps=seed_eps,
        epsilon_agglomerate=epsilon_agglomerate,
        sigma=sigma,
        noise_eps=noise_eps,
        bias=bias,
        filter_fragments=filter_fragments,
        remove_debris=remove_debris,
    )
    run_volara_task(frags_task, blockwise)

    # score RAG edges with waterz
    run_volara_task(
        WaterzAgglom(
            db=db,
            affs_data=affinities,
            frags_data=fragments,
            block_size=block_size,
            context=ctx,
            num_workers=num_workers,
            roi=roi,
            merge_function=waterz_merge_function,
        ),
        blockwise,
    )

    # global segmentation: thresholded connected components -> LUT -> relabel
    graph = db.open("r").read_graph(total_roi, edge_attrs=["merge_score"])
    nodes = np.array(list(graph.nodes), dtype=np.uint64)
    if nodes.size == 0:
        logger.warning("empty RAG; no fragments to agglomerate")
        return

    us, vs, ss = [], [], []
    for u, v, data in graph.edges(data=True):
        score = data.get("merge_score")
        if score is None:  # never merged within threshold range
            continue
        us.append(u)
        vs.append(v)
        ss.append(score)
    edges = (
        np.array(list(zip(us, vs)), dtype=np.uint64)
        if us
        else np.zeros((0, 2), dtype=np.uint64)
    )
    scores = np.array(ss, dtype=np.float32)

    for threshold in thresholds:
        if edges.shape[0] == 0:
            # no scored edges: every fragment is its own segment
            components = nodes.copy()
        else:
            components = connected_components(nodes, edges, scores, threshold)
        name = f"{merge_function}--{threshold}--{shift_name}"
        lut = LUT(path=str(Path(lut_dir) / name))
        lut.save(np.array([nodes, components]))

        run_volara_task(
            Relabel(
                frags_data=fragments,
                seg_data=Labels(store=str(Path(seg_dataset_prefix) / name)),
                lut=lut,
                block_size=block_size,
                roi=roi,
                num_workers=num_workers,
            ),
            blockwise,
        )


def simple_watershed(config):
    import os
    import numpy as np
    from funlib.persistence import open_ds, prepare_ds
    from funlib.geometry import Roi
    from scipy.ndimage import gaussian_filter
    from .ws import watershed_from_affinities
    import waterz

    affs_ds = config["affs_dataset"]
    frags_ds_prefix = config["fragments_dataset"]
    seg_ds_prefix = config["seg_dataset_prefix"]
    mask_ds = config.get("mask_dataset", None)
    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)

    # optional waterz params
    thresholds = config.get("thresholds", [0.2, 0.35, 0.5])
    fragments_in_xy = config.get("fragments_in_xy", True)
    min_seed_distance = config.get("min_seed_distance", 10)
    merge_function = config.get("merge_function", "mean")
    sigma = config.get("sigma", None)
    noise_eps = config.get("noise_eps", None)
    bias = config.get("bias", None)

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
    shift_name = []
    if any([sigma, noise_eps, bias]):
        shift = np.zeros_like(affs_data)

        if noise_eps is not None:
            shift += np.random.randn(*affs_data.shape) * noise_eps
            shift_name.append(f"eps{noise_eps}")

        if sigma is not None:
            shift_name.append(f"sigma{"_".join([str(x) for x in sigma[-3:]])}")
            sigma = (0, *sigma)
            shift += gaussian_filter(affs_data, sigma=sigma) - affs_data

        if bias is not None:
            if type(bias) == float:
                bias = [bias] * affs_data.shape[0]
            else:
                assert len(bias) == affs_data.shape[0]

            shift += np.array([bias]).reshape((-1, *((1,) * (len(affs.shape) - 1))))
            shift_name.append(f"bias{'_'.join([str(x) for x in bias])}")

        affs_data += shift
    shift_name = "--".join(shift_name)

    if affs_data.shape[0] == 2:
        affs_data = np.stack(
            [np.zeros_like(affs_data[0]), affs_data[0], affs_data[1]]
        )

    # watershed
    fragments_data, n = watershed_from_affinities(
        affs_data,
        fragments_in_xy=fragments_in_xy,
        return_seeds=False,
        min_seed_distance=min_seed_distance,
    )

    # write fragments
    shift_name = f"{shift_name}--" if shift_name != "" else ""
    shift_name = f"{shift_name}minseed{min_seed_distance}"
    frags_ds_name = os.path.join(frags_ds_prefix, shift_name)
    frags = prepare_ds(
        frags_ds_name,   
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
        scoring_function=waterz_merge_function,
    )

    for threshold, segmentation in zip(thresholds, generator):
        # write segmentation
        seg_ds_name = os.path.join(seg_ds_prefix, f"{merge_function}--{str(threshold)}--{shift_name}")
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

    if roi_offset is not None and type(roi_offset) == str:
        config["roi_offset"] = list(map(int, roi_offset.strip().split(" ")))
        config["roi_shape"] = list(map(int, roi_shape.strip().split(" ")))

    if blockwise:
        if block_shape == "roi":
            config["blockwise"] = False
        waterz_pipeline(config)
    else:
        simple_watershed(config)
