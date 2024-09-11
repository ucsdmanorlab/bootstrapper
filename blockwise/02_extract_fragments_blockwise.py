import os
import sys
import time
import logging
import yaml
from functools import partial
from pathlib import Path
from pprint import pprint

import numpy as np
import zarr

from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, prepare_ds
from funlib.persistence.graphs import SQLiteGraphDataBase, PgSQLGraphDatabase
from funlib.persistence.types import Vec
import daisy


def watershed_in_block(
        affs, 
        fragments,
        db_config,
        mask_array,
        fragments_in_xy,
        background_mask,
        mask_thresh,
        min_seed_distance,
        epsilon_agglomerate,
        filter_fragments,
        replace_sections, 
        block):
    
    # import
    import logging
    import numpy as np
    import mahotas
    from scipy.ndimage import \
        maximum_filter, \
        gaussian_filter, \
        distance_transform_edt, \
        mean, \
        center_of_mass

    from funlib.persistence import Array
    from funlib.segment.arrays import relabel, replace_values
    from funlib.persistence.graphs import SQLiteGraphDataBase, PgSQLGraphDatabase
    from funlib.persistence.types import Vec
    import waterz

    # watershed functions
    def watershed_from_boundary_distance(
        boundary_distances,
        boundary_mask,
        return_seeds=False,
        id_offset=0,
        min_seed_distance=10,
    ):

        max_filtered = maximum_filter(boundary_distances, min_seed_distance)
        maxima = max_filtered == boundary_distances
        seeds, n = mahotas.label(maxima)

        if n == 0:
            return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

        seeds[seeds != 0] += id_offset

        fragments = mahotas.cwatershed(boundary_distances.max() - boundary_distances, seeds)

        if boundary_mask is not None:
            fragments *= boundary_mask

        ret = (fragments.astype(np.uint64), n + id_offset)
        if return_seeds:
            ret = ret + (seeds.astype(np.uint64),)

        return ret

    def watershed_from_affinities(
        affs,
        max_affinity_value=1.0,
        fragments_in_xy=False,
        background_mask=False,
        mask_thresh=0.5,
        return_seeds=False,
        min_seed_distance=10,
    ):
        """Extract initial fragments from affinities using a watershed
        transform. Returns the fragments and the maximal ID in it.
        Returns:
            (fragments, max_id)
            or
            (fragments, max_id, seeds) if return_seeds == True"""

        # add random noise
        random_noise = np.random.randn(*affs.shape) * 0.01

        # add smoothed affs, to solve a similar issue to the random noise. We want to bias
        # towards processing the central regions of objects first.
        smoothed_affs: np.ndarray = (gaussian_filter(affs, sigma=(0, 1, 2, 2)) - 0.5) * 0.05

        affs = (affs + random_noise + smoothed_affs).astype(np.float32)
        affs = np.clip(affs, 0.0, 1.0)

        if fragments_in_xy:

            mean_affs = 0.5 * (affs[-1] + affs[-2])  # affs are (c,z,y,x)
            depth = mean_affs.shape[0]

            fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
            if return_seeds:
                seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

            id_offset = 0
            for z in range(depth):

                boundary_mask = mean_affs[z] > mask_thresh * max_affinity_value
                boundary_distances = distance_transform_edt(boundary_mask)

                if background_mask is False:
                    boundary_mask = None

                ret = watershed_from_boundary_distance(
                    boundary_distances,
                    boundary_mask,
                    return_seeds=return_seeds,
                    id_offset=id_offset,
                    min_seed_distance=min_seed_distance,
                )

                fragments[z] = ret[0]
                if return_seeds:
                    seeds[z] = ret[2]

                id_offset = ret[1]

            ret = (fragments, id_offset)
            if return_seeds:
                ret += (seeds,)

        else:

            boundary_mask = np.mean(affs, axis=0) > mask_thresh * max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            if background_mask is False:
                boundary_mask = None

            ret = watershed_from_boundary_distance(
                boundary_distances,
                boundary_mask,
                return_seeds,
                min_seed_distance=min_seed_distance,
            )

            fragments = ret[0]

        return ret

    # load data
    affs_data = affs.to_ndarray(block.read_roi)[:3] # short range affinities only

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

    # watershed
    fragments_data, _ = watershed_from_affinities(
        affs_data,
        fragments_in_xy=fragments_in_xy,
        background_mask=background_mask,
        mask_thresh=mask_thresh,
        return_seeds=False,
        min_seed_distance=min_seed_distance,
    )

    # mask fragments
    if mask is not None:
        fragments_data *= mask.astype(np.uint64)

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
        context = (block.write_roi.shape -block.read_roi.shape) / 2

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
        fragments.units
    )

    # crop fragments array to write ROI
    fragments_array = Array(
        fragments_array[block.write_roi],
        block.write_roi.offset,
        affs.voxel_size,
        fragments.axis_names,
        fragments.units
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
    fragments_array.data[fragments_array.data > 0] += id_bump
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

def extract_fragments(config, db_config):

    logging.info("Extracting fragments with config:")
    pprint(config)

    # Extract arguments from config
    affs_file = config["affs_file"] # Path to affinities zarr container
    affs_dataset = config["affs_dataset"] # Name of affinities dataset
    fragments_file = config["fragments_file"] # Path to fragments zarr container
    fragments_dataset = config["fragments_dataset"] # Name of fragments dataset
    num_workers = config["num_workers"] # Number of workers to use

    # Optional parameters
    fragments_in_xy = config.get("fragments_in_xy", True) # Extract fragments for each xy-section separately
    background_mask = config.get("background_mask", False) # Mask out boundaries
    mask_thresh = config.get("mask_thresh", 0.5) # Threshold for boundary mask
    min_seed_distance = config.get("min_seed_distance", 10) # Minimum distance between seeds for watershed
    epsilon_agglomerate = config.get("epsilon_agglomerate", 0) # Perform initial waterz agglomeration
    filter_fragments = config.get("filter_fragments", None) # Filter fragments with average affinity lower than this value
    replace_sections = config.get("replace_sections", None) # Replace fragments data with zero in given sections

    # Read affs
    logging.info(f"Reading affs from {affs_file}/{affs_dataset}")
    affs = open_ds(os.path.join(affs_file, affs_dataset))
    voxel_size = affs.voxel_size

    # get block size, context, ROIs
    if "block_shape" in config and config["block_shape"] is not None:
        block_size = Coordinate(config["block_shape"]) * voxel_size
    else:
        block_size = Coordinate(affs.chunk_shape[1:]) * voxel_size

    if "context" in config and config["context"] is not None:
        context = Coordinate(config["context"]) * voxel_size
    else:
        context = Coordinate([0,] * affs.roi.dims)

    if "roi_offset" in config and "roi_shape" in config:
        roi_offset = config["roi_offset"]
        roi_shape = config["roi_shape"]
    else:
        roi_offset = None
        roi_shape = None

    # get total ROI, read ROI, write ROI
    if roi_offset is not None:
        total_roi = Roi(roi_offset, roi_shape)
    else:
        total_roi = affs.roi

    read_roi = Roi((0,) * affs.roi.dims, block_size).grow(context, context)
    write_roi = Roi((0,) * affs.roi.dims, block_size)
    logging.info(f"Total ROI: {total_roi}, Read ROI: {read_roi}, Write ROI: {write_roi}")
    
    # get mask
    if "mask_file" in config and "mask_dataset" in config and config["mask_file"] is not None:
        mask_array = open_ds(os.path.join(config["mask_file"], config["mask_dataset"]))
    else:
        mask_array = None

    # prepare fragments array
    logging.info(f"Preparing fragments array in {fragments_file}/{fragments_dataset}")
    fragments = prepare_ds(
        store=os.path.join(fragments_file, fragments_dataset),
        shape=total_roi.shape / voxel_size,
        offset=total_roi.offset,
        voxel_size=voxel_size,
        axis_names=affs.axis_names[1:],
        units=affs.units,
        chunk_shape=block_size / voxel_size,
        dtype=np.uint64,
        compressor=zarr.get_codec({"id": "blosc"}),
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
            mask_array,
            fragments_in_xy,
            background_mask,
            mask_thresh,
            min_seed_distance,
            epsilon_agglomerate,
            filter_fragments,
            replace_sections
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

    
if __name__ == "__main__":
    config_file = sys.argv[1] # Path to config file

    # Load config file
    with open(config_file, "r") as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["hglom_segment"]
    db_config = yaml_config["db"]

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    start = time.time()
    extract_fragments(config, db_config)
    end = time.time()

    seconds = end - start
    logging.info(f"Total time to extract fragments: {seconds} ")