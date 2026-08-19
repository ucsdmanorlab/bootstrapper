import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def volara_pipeline(config):
    import os
    from pathlib import Path

    from funlib.geometry import Coordinate
    from funlib.persistence import open_ds
    from volara.blockwise import ExtractFrags, AffAgglom, GraphMWS, Relabel
    from volara.datasets import Affs, Labels, Raw
    from volara.dbs import SQLite, PostgreSQL
    from volara.logging import set_log_basedir
    from volara.lut import LUT

    from .naming import build_name, dump_params, dump_lut_params
    from ..blockwise import run_volara_task

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
    blockwise = config.get("blockwise", False)
    num_workers = config.get("num_workers", 1) if blockwise else 1
    block_shape = config.get("block_shape")
    context = config.get("context")

    if neighborhood is None:
        raise ValueError("Affinities neighborhood must be provided")
    if bias is None:
        raise ValueError("Affinities bias must be provided")
    assert len(neighborhood) == len(
        bias
    ), "Number of biases must match number of affinities channels"

    # per-volume volara logs and done-block caches (CWD-relative by default,
    # which collides across volumes and concurrent runs)
    container = seg_dataset_prefix.rsplit(".zarr", 1)[0] + ".zarr"
    set_log_basedir(
        os.path.join(os.path.dirname(container), f"{Path(container).stem}_volara_logs")
    )

    affs = open_ds(affs_dataset)

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
        block_size = affs.shape[1:]
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
    run_volara_task(extract_frags, blockwise)
    dump_params(frags_ds_name, {"method": "mws", "blockwise": blockwise, **frag_params})

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
    dump_lut_params(lut_name, {"method": "mws", "blockwise": blockwise, **seg_params})

    relabel = Relabel(
        frags_data=fragments,
        seg_data=segments,
        lut=lut,
        block_size=block_size,
        roi=roi,
        num_workers=num_workers * 2,
    )
    run_volara_task(relabel, blockwise)
    dump_params(seg_name, {"method": "mws", "blockwise": blockwise, **seg_params})


def simple_mutex(config):
    import os
    import numpy as np
    from funlib.persistence import open_ds, prepare_ds
    from funlib.geometry import Roi
    from .mws import mwatershed_from_affinities
    from .naming import build_name, dump_params
    from skimage.morphology import remove_small_objects

    affs_ds = config["affs_dataset"]
    frags_ds_prefix = config["fragments_dataset"]
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

    # write fragments; no global_bias here (mwatershed segments in one shot)
    frag_params = {
        "sigma": sigma,
        "noise_eps": noise_eps,
        "bias": bias,
        "strides": strides,
        "randomized_strides": randomized_strides,
    }
    frags_ds_name = os.path.join(frags_ds_prefix, build_name(frag_params))
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
    dump_params(frags_ds_name, {"method": "mws", "blockwise": False, **frag_params})

    # remove small debris
    if remove_debris > 0:
        fragments_dtype = fragments_data.dtype
        fragments_data = fragments_data.astype(np.int64)
        fragments_data = remove_small_objects(fragments_data, min_size=remove_debris)
        fragments_data = fragments_data.astype(fragments_dtype)

    # write segmentation
    seg_params = {**frag_params, "remove_debris": remove_debris}
    seg_ds_name = os.path.join(seg_ds_prefix, build_name(seg_params))
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
    dump_params(seg_ds_name, {"method": "mws", "blockwise": False, **seg_params})


def mutex_watershed_segmentation(config):
    # blockwise or not
    blockwise = config.get("blockwise", False)

    if blockwise:
        if config.get("block_shape") == "roi":
            config["blockwise"] = False
        volara_pipeline(config)
    else:
        simple_mutex(config)
