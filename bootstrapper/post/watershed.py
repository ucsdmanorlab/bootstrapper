import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def watershed_segmentation(config):
    import os
    from importlib.metadata import version
    from pathlib import Path

    import numpy as np
    from funlib.geometry import Coordinate, Roi
    from funlib.persistence import open_ds
    from funlib.segment.graphs.impl import connected_components
    from volara.blockwise import Relabel
    from volara.datasets import Labels, Raw
    from volara.dbs import SQLite, PostgreSQL
    from volara.lut import LUT

    from volara.logging import set_log_basedir

    from .blockwise.watershed_frags import WatershedFrags
    from .blockwise.waterz_agglom import WaterzAgglom, WATERZ_MERGE_FUNCTIONS
    from .naming import build_name, dump_params, dump_lut_params
    from ..blockwise import run_volara_task

    affs_dataset = config["affs_dataset"]
    fragments_dataset_prefix = config["fragments_dataset"]
    seg_dataset_prefix = config["seg_dataset_prefix"]
    lut_dir = config["lut_dir"]
    db_config = config["db"]
    mask_dataset = config.get("mask_dataset")

    # watershed fragment params
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
    if merge_function not in WATERZ_MERGE_FUNCTIONS:
        raise ValueError(
            f"Unknown merge_function '{merge_function}'. Valid values: "
            f"{sorted(WATERZ_MERGE_FUNCTIONS)}. The ZettaAI waterz build only "
            "ships a working 'mean' scorer."
        )
    waterz_merge_function = WATERZ_MERGE_FUNCTIONS[merge_function]

    # blockwise params
    roi_offset = config.get("roi_offset")
    roi_shape = config.get("roi_shape")
    block_shape = config.get("block_shape")
    # a whole-array run is the one-block case
    blockwise = config.get("blockwise", False) and block_shape != "roi"
    num_workers = config.get("num_workers", 1) if blockwise else 1
    context = config.get("context")

    # per-volume volara logs and done-block caches (CWD-relative by default,
    # which collides across volumes and concurrent runs)
    if ".zarr" in seg_dataset_prefix:
        container = seg_dataset_prefix.rsplit(".zarr", 1)[0] + ".zarr"
        log_basedir = os.path.join(
            os.path.dirname(container), f"{Path(container).stem}_volara_logs"
        )
    else:
        # no ".zarr" container to name the logs after
        log_basedir = f"{seg_dataset_prefix}_volara_logs"
    # daisy ships this path to every worker in DAISY_CONTEXT as "key=value"
    # pairs joined by ":", so either character there fails every worker
    if ":" in log_basedir or "=" in log_basedir:
        logger.warning(
            "log dir %s contains ':' or '='; keeping the default volara log dir",
            log_basedir,
        )
    else:
        set_log_basedir(log_basedir)

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

    frag_params = {
        "fragments_in_xy": fragments_in_xy,
        "min_seed_distance": min_seed_distance,
        "seed_eps": seed_eps,
        "epsilon_agglomerate": epsilon_agglomerate,
        "sigma": sigma,
        "noise_eps": noise_eps,
        "bias": bias,
        "filter_fragments": filter_fragments,
        "remove_debris": remove_debris,
    }
    shift_name = build_name(frag_params)
    frags_ds_name = str(Path(fragments_dataset_prefix) / shift_name)

    # recorded on every output: the inputs and the region a name cannot show
    run_params = {
        "method": "ws",
        "blockwise": blockwise,
        "affs_dataset": affs_dataset,
        "mask_dataset": mask_dataset,
        "aff_neighborhood": config.get("aff_neighborhood"),
        "roi_offset": list(total_roi.offset),
        "roi_shape": list(total_roi.shape),
        "block_shape": list(block_size),
        "context": list(ctx),
        "bootstrapper_version": version("bootstrapper"),
    }

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
    dump_params(frags_ds_name, {**run_params, **frag_params})

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
        params = {"merge_function": merge_function, "threshold": threshold, **frag_params}
        name = build_name(params)
        recorded = {**run_params, **params}

        lut = LUT(path=str(Path(lut_dir) / name))
        lut.save(np.array([nodes, components]))
        dump_lut_params(str(Path(lut_dir) / name), recorded)

        seg_store = str(Path(seg_dataset_prefix) / name)
        run_volara_task(
            Relabel(
                frags_data=fragments,
                seg_data=Labels(store=seg_store),
                lut=lut,
                block_size=block_size,
                roi=roi,
                num_workers=num_workers,
            ),
            blockwise,
        )
        dump_params(seg_store, recorded)
