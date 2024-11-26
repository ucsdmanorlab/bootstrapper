import click
import time
import logging
import toml
from functools import partial
from pathlib import Path
import os
from pprint import pprint

import numpy as np
import zarr

from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, prepare_ds
from funlib.persistence.graphs import SQLiteGraphDataBase, PgSQLGraphDatabase
from funlib.persistence.types import Vec
import daisy

logging.basicConfig(level=logging.INFO)


def watershed_in_block(
    affs,
    fragments,
    db_config,
    shift,
    mask_array,
    fragments_in_xy,
    min_seed_distance,
    epsilon_agglomerate,
    filter_fragments,
    replace_sections,
    block,
):

    # import
    import numpy as np
    from scipy.ndimage import mean, center_of_mass, gaussian_filter

    from funlib.persistence import Array
    from funlib.segment.arrays import relabel, replace_values
    from funlib.persistence.graphs import SQLiteGraphDataBase, PgSQLGraphDatabase
    from funlib.persistence.types import Vec
    import waterz

    from bootstrapper.post.ws import watershed_from_affinities

    # load data
    affs_data = affs[block.read_roi][:3]

    # normalize
    if affs_data.dtype == np.uint8:
        affs_data = affs_data.astype(np.float32) / 255.0
    else:
        affs_data = affs_data.astype(np.float32)

    # load mask
    if mask_array is not None:
        mask = mask_array.to_ndarray(block.read_roi)
    else:
        mask = None

    # shift affs with noise, smoothing, and bias
    if shift is not None:
        sigma = shift["sigma"]
        noise_eps = shift["noise_eps"]
        bias = shift["bias"]

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

            shift += np.array([bias]).reshape((-1, *((1,) * (len(affs.shape) - 1))))

        affs_data += shift

    # mask affinities
    if mask is not None:
        affs_data *= (mask > 0).astype(np.uint8)

    # watershed
    fragments_data, n = watershed_from_affinities(
        affs_data,
        fragments_in_xy=fragments_in_xy,
        return_seeds=False,
        min_seed_distance=min_seed_distance,
    )

    logging.info(f"Found {n} fragments in block {block}")

    # filter fragments
    if filter_fragments is not None and filter_fragments > 0:
        mean_affs = np.mean(affs_data, axis=0)

        filtered_fragments = []

        fragment_ids = np.unique(fragments_data)

        for fragment, mean_aff_value in zip(
            fragment_ids, mean(mean_affs, fragments_data, fragment_ids)
        ):
            if mean_aff_value < filter_fragments:
                filtered_fragments.append(fragment)

        filtered_fragments = np.array(filtered_fragments, dtype=fragments_data.dtype)
        replace = np.zeros_like(filtered_fragments)
        replace_values(fragments_data, filtered_fragments, replace, inplace=True)

    # epsilon agglomerate
    if epsilon_agglomerate > 0:

        # add fake z-affinity channel if 2D affinities
        if affs.shape[0] == 2:
            affs.data = np.stack(
                [np.zeros_like(affs.data[0]), affs.data[-2], affs.data[-1]]
            )

        generator = waterz.agglomerate(
            affs=affs_data,
            thresholds=[epsilon_agglomerate],
            fragments=fragments_data,
            # scoring_function='OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
            scoring_function="OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
            discretize_queue=256,
            return_merge_history=False,
            return_region_graph=False,
        )
        fragments_data[:] = next(generator)

        # cleanup generator
        for _ in generator:
            pass

    # replace sections
    if replace_sections is not None:
        block_begin = block.write_roi.get_begin()
        shape = block.write_roi.get_shape()
        context = (block.write_roi.shape - block.read_roi.shape) / 2

        z_context = context[0] / affs.voxel_size[0]

        mapping = {}

        voxel_offset = block_begin[0] / affs.voxel_size[0]

        for i, j in zip(range(fragments_data.shape[0]), range(shape[0])):
            mapping[i] = i
            mapping[j] = (
                int(voxel_offset + i)
                if block_begin[0] == affs.roi.get_begin()[0]
                else int(voxel_offset + (i - z_context))
            )

        replace = [k for k, v in mapping.items() if v in replace_sections]

        for r in replace:
            fragments_data[r] = 0

    # create fragments array
    fragments_array = Array(
        fragments_data,
        block.read_roi.offset,
        affs.voxel_size,
        fragments.axis_names,
        fragments.units,
    )

    # crop fragments array to write ROI
    fragments_array = Array(
        fragments_array[block.write_roi],
        block.write_roi.offset,
        affs.voxel_size,
        fragments.axis_names,
        fragments.units,
    )
    fragments_array.data = fragments_array.data.compute()

    # ensure unique fragment ids
    max_id = fragments_array.data.max()
    num_voxels_in_block = np.prod(fragments_array.data.shape)
    if max_id > num_voxels_in_block:
        fragments_array.data, max_id = relabel(fragments_array.data)
        assert max_id < num_voxels_in_block

    # bump fragment IDs
    id_bump = block.block_id[1] * num_voxels_in_block
    fragments_array.data[fragments_array.data > 0] += int(id_bump)
    fragment_ids = range(id_bump + 1, id_bump + 1 + int(max_id))

    # store fragments
    fragments[block.write_roi] = fragments_array.data

    # exit if no fragments were found
    if max_id == 0:
        return 0

    # load RAG DB
    if "db_file" in db_config:
        # SQLiteGraphDatabase
        rag_provider = SQLiteGraphDataBase(
            db_file=Path(db_config["db_file"]),
            position_attribute="center",
            mode="r+",
            nodes_table=db_config["nodes_table"],
            edges_table=db_config["edges_table"],
            node_attrs={"center": Vec(int, affs.roi.dims)},
            edge_attrs={"merge_score": float, "agglomerated": bool},
        )
    else:
        # PgSQLGraphDatabase
        rag_provider = PgSQLGraphDatabase(
            position_attribute="center",
            db_name=db_config["db_name"],
            db_host=db_config["db_host"],
            db_user=db_config["db_user"],
            db_password=db_config["db_password"],
            db_port=db_config["db_port"],
            mode="r+",
            nodes_table=db_config["nodes_table"],
            edges_table=db_config["edges_table"],
            node_attrs={"center": Vec(int, affs.roi.dims)},
            edge_attrs={"merge_score": float, "agglomerated": bool},
        )

    # get fragment centers
    fragment_centers = {
        fragment: block.write_roi.get_offset() + affs.voxel_size * Coordinate(center)
        for fragment, center in zip(
            fragment_ids,
            center_of_mass(fragments_array.data, fragments_array.data, fragment_ids),
        )
        if not np.isnan(center[0])
    }

    # store centers as nodes in rag
    rag = rag_provider[block.write_roi]
    rag.add_nodes_from([(node, {"center": c}) for node, c in fragment_centers.items()])
    rag_provider.write_graph(rag, block.write_roi)

    return 0


def extract_fragments(config):

    logging.info("Extracting fragments with config:")

    # Extract arguments from config
    affs_dataset = config["affs_dataset"]  # Name of affinities dataset
    fragments_dataset_prefix = config["fragments_dataset"]  # Name of fragments dataset
    db_config = config["db"]  # Database configuration

    # Optional parameters
    roi_offset = config.get("roi_offset", None)  # Offset of ROI
    roi_shape = config.get("roi_shape", None)  # Shape of ROI
    blockwise = config.get("blockwise", False)  # Perform blockwise extraction
    num_workers = config.get("num_workers", 1)  # Number of workers to use
    block_shape = config.get("block_shape", None)  # Shape of block
    context = config.get("context", None)  # Context for block

    # optional watershed parameters
    fragments_in_xy = config.get(
        "fragments_in_xy", True
    )  # Extract fragments for each xy-section separately
    min_seed_distance = config.get(
        "min_seed_distance", 10
    )  # Minimum distance between seeds for watershed
    epsilon_agglomerate = config.get(
        "epsilon_agglomerate", 0
    )  # Perform initial waterz agglomeration
    filter_fragments = config.get(
        "filter_fragments", None
    )  # Filter fragments with average affinity lower than this value
    replace_sections = config.get(
        "replace_sections", None
    )  # Replace fragments data with zero in given sections

    sigma = config.get("sigma", None)
    noise_eps = config.get("noise_eps", None)
    bias = config.get("bias", None)

    shift_name = []
    if sigma is not None or noise_eps is not None or bias is not None:
        affs_shift = {
            "sigma": sigma,
            "noise_eps": noise_eps,
            "bias": bias,
        }
        if noise_eps is not None:
            shift_name.append(f"{noise_eps}")
        if sigma is not None:
            shift_name.append(f"{"_".join([str(x) for x in sigma[-3:]])}")
        if bias is not None:
            shift_name.append(f"{"_".join([str(x) for x in bias])}")
    else:
        affs_shift = None

    shift_name = "--".join(shift_name)

    # Read affs
    logging.info(f"Reading affs from {affs_dataset}")
    affs = open_ds(affs_dataset)
    voxel_size = affs.voxel_size

    # get total ROI
    if roi_offset is not None:
        total_roi = Roi(roi_offset, roi_shape)
    else:
        total_roi = affs.roi

    # get block size, context
    if blockwise:
        if block_shape is not None:
            block_size = Coordinate(block_shape) * voxel_size
        else:
            block_size = Coordinate(affs.chunk_shape[1:]) * voxel_size

        if context is not None:
            context = Coordinate(context) * voxel_size
        else:
            context = Coordinate(
                [
                    0,
                ]
                * affs.roi.dims
            )

    else:  # blockwise is False
        block_size = total_roi.get_shape()
        context = Coordinate(
            [
                0,
            ]
            * affs.roi.dims
        )
        num_workers = 1

    # get block read ROI, write ROI
    read_roi = Roi((0,) * affs.roi.dims, block_size).grow(context, context)
    write_roi = Roi((0,) * affs.roi.dims, block_size)
    logging.info(
        f"Total ROI: {total_roi}, Read ROI: {read_roi}, Write ROI: {write_roi}"
    )

    # get mask
    if "mask_dataset" in config and config["mask_dataset"] is not None:
        mask_array = open_ds(config["mask_dataset"])
    else:
        mask_array = None

    # prepare fragments array
    shift_name = f"{shift_name}--" if shift_name != "" else ""
    shift_name = f"{shift_name}minseed{min_seed_distance}"
    frags_ds_name = os.path.join(fragments_dataset_prefix, shift_name)
    logging.info(f"Preparing fragments array in {frags_ds_name}")
    fragments = prepare_ds(
        store=frags_ds_name,
        shape=total_roi.shape / voxel_size,
        offset=total_roi.offset,
        voxel_size=voxel_size,
        axis_names=affs.axis_names[1:],
        units=affs.units,
        chunk_shape=block_size / voxel_size if blockwise else None,
        dtype=np.uint64,
        compressor=zarr.get_codec({"id": "blosc"}),
        mode="w",
    )

    # prepare RAG provider
    logging.info(msg="Creating RAG DB...")
    if "db_file" in db_config:
        # SQLiteGraphDatabase
        rag_provider = SQLiteGraphDataBase(
            position_attribute="center",
            db_file=Path(db_config["db_file"]),
            mode="w",
            nodes_table=db_config["nodes_table"],
            edges_table=db_config["edges_table"],
            node_attrs={"center": Vec(int, affs.roi.dims)},
            edge_attrs={"merge_score": float, "agglomerated": bool},
        )
        rag_provider.con.close()

    else:
        # PgSQLGraphDatabase
        rag_provider = PgSQLGraphDatabase(
            position_attribute="center",
            db_name=db_config["db_name"],
            db_host=db_config["db_host"],
            db_user=db_config["db_user"],
            db_password=db_config["db_password"],
            db_port=db_config["db_port"],
            mode="w",
            nodes_table=db_config["nodes_table"],
            edges_table=db_config["edges_table"],
            node_attrs={"center": Vec(int, affs.roi.dims)},
            edge_attrs={"merge_score": float, "agglomerated": bool},
        )
        rag_provider.connection.close()

    # prepare blockwise task
    task = daisy.Task(
        "ExtractFragments",
        total_roi.grow(context, context),
        read_roi,
        write_roi,
        process_function=partial(
            watershed_in_block,
            affs,
            fragments,
            db_config,
            affs_shift,
            mask_array,
            fragments_in_xy,
            min_seed_distance,
            epsilon_agglomerate,
            filter_fragments,
            replace_sections,
        ),
        read_write_conflict=False,
        num_workers=num_workers,
        max_retries=5,
        fit="shrink",
    )

    # Run blockwise
    ret = daisy.run_blockwise([task])

    if ret:
        print("Ran all blocks successfully!")
    else:
        print("Did not run all blocks successfully...")

    return frags_ds_name


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def frags(config_file):
    """
    Extract fragments from affinities using daisy.
    """

    # Load config file
    with open(config_file, "r") as f:
        config = toml.load(f)

    start = time.time()
    extract_fragments(config)
    end = time.time()

    seconds = end - start
    logging.info(f"Total time to extract fragments: {seconds} ")
