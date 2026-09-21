import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mutex_watershed_segmentation(config):
    import os
    from pathlib import Path

    from funlib.geometry import Coordinate
    from funlib.persistence import open_ds
    from volara.blockwise import AffAgglom, GraphMWS, Relabel
    from volara.datasets import Affs, Labels, Raw
    from volara.dbs import SQLite, PostgreSQL
    from volara.logging import set_log_basedir
    from volara.lut import LUT

    from .blockwise.extract_frags import ExtractFrags
    from .naming import build_name, dump_params, dump_lut_params, inputs_differ
    from ..blockwise import run_volara_task, volara_log_dir

    affs_dataset = config["affs_dataset"]
    fragments_dataset_prefix = config["fragments_dataset"]
    db_config = config["db"]
    mask_dataset = config.get("mask_dataset")
    lut_dir = config["lut_dir"]
    seg_dataset_prefix = config["seg_dataset_prefix"]

    # required mws params
    neighborhood = config.get("aff_neighborhood")
    bias = config.get("bias")
    global_bias = tuple(config.get("global_bias", [1.0, -0.5]))

    # optional mws params
    filter_fragments = config.get("filter_fragments")
    sigma = config.get("sigma")
    noise_eps = config.get("noise_eps")
    strides = config.get("strides")
    randomized_strides = config.get("randomized_strides", False)
    remove_debris = config.get("remove_debris", 0)
    min_seed_distance = config.get("min_seed_distance")

    # blockwise params
    roi_offset = config.get("roi_offset")
    roi_shape = config.get("roi_shape")
    block_shape = config.get("block_shape")
    # a whole-array run is the one-block case
    blockwise = config.get("blockwise", False) and block_shape != "roi"
    num_workers = config.get("num_workers", 1) if blockwise else 1
    context = config.get("context")

    if neighborhood is None:
        raise ValueError("Affinities neighborhood must be provided")
    if bias is None:
        raise ValueError("Affinities bias must be provided")
    if len(neighborhood) != len(bias):
        raise ValueError(
            f"{len(bias)} biases for {len(neighborhood)} neighborhood offsets: give one per offset"
        )

    set_log_basedir(volara_log_dir(seg_dataset_prefix))

    affs = open_ds(affs_dataset)
    if affs.shape[0] != len(neighborhood):
        raise ValueError(
            f"{affs_dataset} has {affs.shape[0]} affinity channels but the neighborhood "
            f"has {len(neighborhood)} offsets: use a neighborhood with one offset per channel"
        )

    if roi_offset is not None:
        roi = (roi_offset, roi_shape)
    else:
        roi = (affs.roi.offset, affs.roi.shape)

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
        block_size = Coordinate(roi[1]) / affs.voxel_size
        ctx = Coordinate([0] * affs.roi.dims)

    # dataset names: frags from fragment params; lut/seg add the global mws
    # bias so runs differing only in global_bias cannot clobber each other
    frag_params = {
        "min_seed_distance": min_seed_distance,
        "sigma": sigma,
        "noise_eps": noise_eps,
        "bias": bias,
        "strides": strides,
        "randomized_strides": randomized_strides,
        "filter_fragments": filter_fragments,
        "remove_debris": remove_debris,
    }
    seg_params = {"global_bias": list(global_bias), **frag_params}
    shift_name = build_name(frag_params)
    agglom_name = build_name(seg_params)
    frags_ds_name = str(Path(fragments_dataset_prefix) / shift_name)
    lut_name = str(Path(lut_dir) / agglom_name)
    seg_name = str(Path(seg_dataset_prefix) / agglom_name)

    # recorded on every output: the inputs and the region a name cannot show
    run_params = {
        "method": "mws",
        "blockwise": blockwise,
        "affs_dataset": affs_dataset,
        "mask_dataset": mask_dataset,
        "aff_neighborhood": neighborhood,
        "roi_offset": list(roi[0]),
        "roi_shape": list(roi[1]),
        "block_shape": list(block_size),
        "context": list(ctx),
    }

    affinities = Affs(store=affs_dataset, neighborhood=neighborhood)
    mask_data = Raw(store=mask_dataset) if mask_dataset else None
    if "db_file" in db_config:
        db = SQLite(path=db_config["db_file"], edge_attrs={"zyx_aff": "float"})
    else:
        db = PostgreSQL(
            name=db_config["db_name"],
            host=db_config["db_host"],
            user=db_config["db_user"],
            password=db_config["db_password"],
            edge_attrs={"zyx_aff": "float"},
        )
    fragments = Labels(store=frags_ds_name)
    segments = Labels(store=seg_name)
    os.makedirs(lut_dir, exist_ok=True)
    lut = LUT(path=lut_name)

    extract_frags = ExtractFrags(
        db=db,
        affs_data=affinities,
        frags_data=fragments,
        mask_data=mask_data,
        block_size=block_size,
        context=ctx,
        num_workers=num_workers,
        roi=roi,
        bias=bias,
        sigma=sigma,
        noise_eps=noise_eps,
        filter_fragments=filter_fragments,
        remove_debris=remove_debris,
        strides=strides,
        randomized_strides=randomized_strides,
        min_seed_distance=min_seed_distance,
    )
    warning = inputs_differ(frags_ds_name, run_params)
    if warning:
        logger.warning(warning)
    run_volara_task(extract_frags, blockwise)
    dump_params(frags_ds_name, {**run_params, **frag_params})

    aff_agglom = AffAgglom(
        db=db,
        affs_data=affinities,
        frags_data=fragments,
        block_size=block_size,
        context=ctx,
        scores={"zyx_aff": affinities.neighborhood},
        num_workers=num_workers,
        roi=roi,
    )
    run_volara_task(aff_agglom, blockwise)

    global_mws = GraphMWS(
        db=db,
        lut=lut,
        weights={"zyx_aff": global_bias},
        roi=roi,
    )
    run_volara_task(global_mws, multiprocessing=False)
    dump_lut_params(lut_name, {**run_params, **seg_params})

    relabel = Relabel(
        frags_data=fragments,
        seg_data=segments,
        lut=lut,
        block_size=block_size,
        roi=roi,
        num_workers=num_workers * 2,
    )
    run_volara_task(relabel, blockwise)
    dump_params(seg_name, {**run_params, **seg_params})
