import time
import logging
from functools import partial

import click
import numpy as np
from tqdm import tqdm
import daisy
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, prepare_ds
import fastremap
import cc3d

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.group()
def refine():
    """Refine segmented volumes: dust/thin removal, size-outlier removal, hole filling."""
    pass


def filter_array(labels_array, to_remove=None, to_merge=None, size=16, min_z_per_label=1):
    merge_mapping = {}
    # Remove given ids
    if to_remove is not None:
        for rid in to_remove:
            merge_mapping[rid] = 0

    # Merge given ids
    if to_merge is not None:
        for merge_ids in to_merge:
            for mid in merge_ids:
                merge_mapping[mid] = merge_ids[0]

    if len(merge_mapping) > 0:
        st = time.time()
        print(f"pre-Mapping {len(merge_mapping)} user-given ids: {merge_mapping}")
        labels_array = fastremap.remap(labels_array, merge_mapping, preserve_missing_labels=True)
        print(f"pre-mapping time: {time.time() - st}")

    # cc3d
    st = time.time()
    labels_array = cc3d.connected_components(labels_array, connectivity=6)
    print(f"Time to cc3d: {time.time() - st}")

    # get contacting voxels
    st = time.time()
    contacts = cc3d.contacts(labels_array, connectivity=6, surface_area=False)
    print(f"Time to get contacts: {time.time() - st}")

    # get label statistics
    st = time.time()
    stats = cc3d.statistics(labels_array, no_slice_conversion=True)
    counts = stats['voxel_counts']
    bboxes = stats['bounding_boxes']
    print(f"Time to get label statistics: {time.time() - st}")

    mapping = {}
    st = time.time()
    for label, _ in tqdm(cc3d.each(labels_array)):
        count = counts[label]
        num_z = bboxes[label][1] - bboxes[label][0]
        contact = {u[u.index(label) ^ 1]:v for u,v in contacts.items() if label in u}

        if num_z < min_z_per_label or count <= size:
            if contact == {}:
                mapping[label] = 0
            elif max(contact.values()) <= size // 2:
                mapping[label] = 0
            else:
                biggest_neighbor = max(contact, key=contact.get)
                mapping[label] = biggest_neighbor

    print(f"Time to create mapping: {time.time() - st}")
    st = time.time()
    labels_array = fastremap.remap(labels_array, mapping, preserve_missing_labels=True, in_place=True)
    print(f"Time to map labels: {time.time() - st}")

    return labels_array


@refine.command("filter")
@click.option(
    "--in_array",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="The path of the input zarr array",
)
@click.option(
    "--out_array",
    "-o",
    type=click.Path(),
    help="The path of the output mask zarr array",
)
@click.option(
    "--size",
    "-s",
    type=int,
    default=100,
    help="Size for dust removal",
)
@click.option(
    "--min_z_per_label",
    "-z",
    type=int,
    default=1,
    help="Minimum number of z-slices per label, labels existing for this many z-slices or less are removed",
)
@click.option(
    "--remove_ids",
    "-r",
    type=str,
    default=None,
    help="Comma-separated list of label IDs to remove",
)
@click.option(
    "--merge_ids",
    "-m",
    type=str,
    multiple=True,
    default=list(),
    help="Comma-separated list of label IDs to merge into a single label",
)
def filter_labels(in_array, out_array, size, min_z_per_label, remove_ids, merge_ids):
    """
    Remove small/thin objects and unwanted IDs from segmented volumes.

    Loads the full array into RAM (cc3d needs a global relabel). Use this for
    dust removal, min-z filtering, and user remove/merge on volumes that fit.

    Args:
        in_array (str): Path to the input zarr array.
        out_array (str): Path to the output zarr array.
        size (int): Size for dust removal.
        min_z_per_label (int): Minimum number of z-slices per label.
        remove_ids (list): List of label IDs to remove.
        merge_ids (list): List of label IDs to merge.

    Returns:
        str: Path to the output array.
    """

    if out_array is None:
        in_f, in_ds_name = in_array.split(".zarr/")
        out_ds =  in_ds_name + "_filtered"
        out_array = f"{in_f}.zarr/{out_ds}"

    in_labels = open_ds(in_array)
    total_roi = in_labels.roi

    out_labels = prepare_ds(
        out_array,
        shape=total_roi.shape / in_labels.voxel_size,
        offset=total_roi.offset,
        voxel_size=in_labels.voxel_size,
        axis_names=in_labels.axis_names,
        units=in_labels.units,
        dtype=in_labels.dtype,
        chunk_shape=in_labels.chunk_shape,
        mode="w",
    )

    to_remove = [int(x) for x in remove_ids.replace(' ','').split(",")] if remove_ids is not None else None
    to_merge = [[int(x) for x in merge_set.replace(' ','').split(",")] for merge_set in merge_ids] if len(merge_ids) > 0 else None

    # read
    st = time.time()
    labels_array = in_labels[total_roi]
    print(f"Time to read labels: {time.time() - st}")

    # process
    st = time.time()
    labels_array = filter_array(labels_array, to_remove=to_remove, to_merge=to_merge, size=size, min_z_per_label=min_z_per_label)

    # write
    out_labels[total_roi] = labels_array
    print(f"Total time to filter labels: {time.time() - st}")


def _remove_ids_blockwise(in_ds, out_ds, remove_ids, block):
    data = in_ds.to_ndarray(block.read_roi)
    if remove_ids.size > 0:
        fastremap.mask(data, remove_ids, in_place=True)  # listed ids -> 0
    out_ds[block.write_roi] = data
    return 0


@refine.command("outliers")
@click.option(
    "--in_array",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="The path of the input zarr array (final, globally-labelled segmentation)",
)
@click.option(
    "--out_array",
    "-o",
    type=click.Path(),
    help="The path of the output zarr array",
)
@click.option(
    "--num_std",
    "-n",
    type=float,
    default=5.0,
    help="Remove objects larger than mean + num_std * std of object size",
)
@click.option(
    "--min_size",
    type=int,
    default=0,
    help="Objects smaller than this are excluded from the mean/std so debris "
    "does not drag the cutoff down and delete legitimate large cells. Set it "
    "around the dust threshold. These small objects are never removed here.",
)
@click.option(
    "--max_size",
    type=int,
    default=None,
    help="Absolute voxel-count cap; if given, overrides the statistical cutoff",
)
@click.option(
    "--num_workers",
    "-w",
    type=int,
    default=20,
    help="Number of workers for the blockwise removal pass",
)
@click.option(
    "--dry_run",
    is_flag=True,
    default=False,
    help="Compute and print the cutoff and objects to remove without writing",
)
def filter_outliers(in_array, out_array, num_std, min_size, max_size, num_workers, dry_run):
    """
    Remove ultra-large object-size outliers, blockwise.

    One-sided upper cut only: removes objects whose voxel count exceeds
    mean + num_std * std (over objects >= min_size), or --max_size. Tiny
    objects are left to the dust removal in the `filter` command. Object
    identity is the global MWS id (no per-block cc3d), so the two passes never
    hold the whole array in RAM.
    """

    in_ds = open_ds(in_array)
    voxel_size = in_ds.voxel_size
    offset = in_ds.roi.offset
    nz, ny, nx = in_ds.shape
    chunk = in_ds.chunk_shape

    # chunk-aligned XY tiles (~4096 voxels), spanning all z
    chunk_xy = int(chunk[-1])
    tile = chunk_xy * max(1, round(4096 / chunk_xy))

    # Pass 1: accumulate global per-id voxel counts (read-only, one tile resident)
    st = time.time()
    us, cs = [], []
    for iy in range(0, ny, tile):
        for ix in range(0, nx, tile):
            sy, sx = min(tile, ny - iy), min(tile, nx - ix)
            roi = Roi(offset + Coordinate(0, iy, ix) * voxel_size,
                      Coordinate(nz, sy, sx) * voxel_size)
            u, c = fastremap.unique(in_ds.to_ndarray(roi), return_counts=True)
            us.append(u)
            cs.append(c)
    labels = np.concatenate(us)
    counts = np.concatenate(cs)
    uniq, inv = np.unique(labels, return_inverse=True)
    sizes = np.bincount(inv, weights=counts.astype(np.float64)).astype(np.int64)
    fg = uniq != 0
    uniq, sizes = uniq[fg], sizes[fg]
    print(f"pass 1: {uniq.size} objects counted in {time.time() - st:.1f}s")

    # statistics over "real" objects only: debris below min_size would drag
    # the mean/std down and push the cutoff below legitimate large cells.
    stat_sizes = sizes[sizes >= min_size]
    if stat_sizes.size == 0:
        raise ValueError(f"no objects with size >= min_size ({min_size})")
    mean, std = float(stat_sizes.mean()), float(stat_sizes.std())
    cutoff = float(max_size) if max_size is not None else mean + num_std * std
    remove_ids = uniq[sizes > cutoff].astype(uniq.dtype)  # upper cut only

    p50, p90, p99, p999 = np.percentile(stat_sizes, [50, 90, 99, 99.9])
    print(f"objects >= min_size({min_size}): {stat_sizes.size} of {uniq.size}")
    print(f"size distribution: p50={p50:.0f} p90={p90:.0f} p99={p99:.0f} "
          f"p99.9={p999:.0f} max={int(sizes.max())}")
    print(f"mean={mean:.1f} std={std:.1f} cutoff={cutoff:.1f} "
          f"-> removing {remove_ids.size} objects")

    if dry_run:
        print("dry run; nothing written")
        return

    if out_array is None:
        in_f, in_ds_name = in_array.split(".zarr/")
        out_array = f"{in_f}.zarr/{in_ds_name}_outliers"

    print(f"Writing to {out_array}")
    out_ds = prepare_ds(
        out_array,
        shape=in_ds.shape,
        offset=offset,
        voxel_size=voxel_size,
        axis_names=in_ds.axis_names,
        units=in_ds.units,
        dtype=in_ds.dtype,
        chunk_shape=chunk,
        mode="w",
    )

    # Pass 2: blockwise removal (chunk-aligned, disjoint writes, no context)
    dims = in_ds.roi.dims
    write_block_roi = daisy.Roi((0,) * dims, Coordinate(nz, tile, tile) * voxel_size)
    task = daisy.Task(
        "FilterOutliersTask",
        out_ds.roi,
        write_block_roi,
        write_block_roi,
        process_function=partial(_remove_ids_blockwise, in_ds, out_ds, remove_ids),
        read_write_conflict=False,
        num_workers=num_workers,
        max_retries=0,
        fit="shrink",
    )
    ret = daisy.run_blockwise([task])
    print("Ran all blocks successfully!" if ret else "Did not run all blocks successfully...")
    return out_array


def fill_holes_blockwise(in_ds, out_ds, block):
    import numpy as np
    from funlib.persistence import Array
    import fastmorph

    logger.info(f"Filling holes for block: {block.write_roi}")

    # clip the halo at the true volume edge: reading beyond the array and
    # zero-padding would make fix_borders treat that fake background as a
    # border and erode real voxels next to it. Interior block seams keep their
    # halo (real neighbor data); the true border stays a genuine cutout edge,
    # matching the old whole-plane behavior.
    read_roi = block.read_roi.intersect(in_ds.roi)
    in_data = in_ds.to_ndarray(read_roi)
    out_data = in_data.copy()

    # Only 6 physical z-sections: a true 3D fill leaves any gap that is open
    # along z (a through-section tube, or a hole present only in an interior
    # section) unfilled, because such background touches a z-face and is treated
    # as exterior. Fill each section in 2D instead. fastmorph has no 2D fill, so
    # stack a section 3x in z, fill in 3D with fix_borders, and keep the middle.
    for z in range(in_data.shape[0]):
        section = in_data[z : z + 1]
        stacked = np.repeat(section, 3, axis=0)
        closed = fastmorph.closing(stacked, parallel=2)
        closed = fastmorph.spherical_close(closed, radius=1, parallel=2)
        filled = fastmorph.fill_holes_v2(
            closed, merge_threshold=0.95, fix_borders=True, parallel=2
        )[0]
        out_data[z] = filled[1]

    # write only the interior write_roi; the halo (and any hole that reached the
    # read edge and was filled inconsistently) is discarded
    out_array = Array(out_data, read_roi.offset, out_ds.voxel_size)
    out_ds[block.write_roi] = out_array.to_ndarray(block.write_roi)

    return 0


@refine.command("fill-holes")
@click.option(
    "--in_array",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="The path of the input zarr array",
)
@click.option(
    "--out_array",
    "-o",
    type=click.Path(),
    help="The path of the output mask zarr array",
)
@click.option(
    "--num_workers",
    "-w",
    type=int,
    default=20,
    help="Number of workers for parallel processing",
)
@click.option(
    "--block_size",
    "-b",
    type=int,
    default=1024,
    help="XY tile size in voxels (all z-sections are kept in every block)",
)
@click.option(
    "--context",
    "-c",
    type=int,
    default=128,
    help="XY halo in voxels. A background gap is only filled if it does not "
    "reach the block's read edge, so context must be >= the largest interior "
    "hole's XY extent. A hole larger than the halo reaches the read edge in "
    "every block covering it and is left UNFILLED (measure the max hole size "
    "and set context above it).",
)
@click.option(
    "--chunk_size",
    type=int,
    default=512,
    help="XY chunk size of the output; block_size must be a multiple of it",
)
def fill_holes(in_array, out_array, num_workers, block_size, context, chunk_size):
    """
    Fill interior gaps in segmented volumes, blockwise (XY-tiled).

    Args:
        in_array (str): Path to the input zarr array.
        out_array (str): Path to the output zarr array.
        num_workers (int): Number of parallel workers.
        block_size (int): XY tile size in voxels.
        context (int): XY halo in voxels.
        chunk_size (int): XY chunk size of the output array.

    Returns:
        str: Path to the output array.
    """

    if block_size % chunk_size != 0:
        raise ValueError(
            f"block_size ({block_size}) must be a multiple of chunk_size "
            f"({chunk_size}) so blocks never share a zarr chunk"
        )

    # open
    in_ds = open_ds(in_array)
    dims = in_ds.roi.dims
    num_z = in_ds.shape[0]
    voxel_size = in_ds.voxel_size

    # blocks span all z and tile XY; halo is XY-only
    write_shape = daisy.Coordinate((num_z, block_size, block_size))
    context_shape = daisy.Coordinate((0, context, context))
    write_block_roi = daisy.Roi((0,) * dims, write_shape * voxel_size)
    read_block_roi = write_block_roi.grow(
        context_shape * voxel_size, context_shape * voxel_size
    )

    if out_array is None:
        in_f, in_ds_name = in_array.split(".zarr/")
        out_ds_name = in_ds_name + "_filled"
        out_array = f"{in_f}.zarr/{out_ds_name}"

    print(f"Writing filled labels to {out_array}")
    out_ds = prepare_ds(
        out_array,
        shape=in_ds.shape,
        offset=in_ds.roi.offset,
        voxel_size=voxel_size,
        axis_names=in_ds.axis_names,
        units=in_ds.units,
        dtype=in_ds.dtype,
        chunk_shape=daisy.Coordinate((num_z, chunk_size, chunk_size)),
    )

    # run: grow total_roi by context so the write blocks (offset inward by
    # context within their read block) cover the full array, including the
    # outer context-wide border. Without the grow, daisy leaves that border
    # strip unwritten (it stays at the fill value 0). The read halo that runs
    # off the true edge is clipped in the worker. in_ds != out_ds and writes
    # are chunk-aligned and disjoint, so there is no read/write conflict.
    context_world = context_shape * voxel_size
    task = daisy.Task(
        "FillHolesTask",
        out_ds.roi.grow(context_world, context_world),
        read_block_roi,
        write_block_roi,
        process_function=partial(fill_holes_blockwise, in_ds, out_ds),
        read_write_conflict=False,
        num_workers=num_workers,
        max_retries=0,
        fit="shrink",
    )

    ret = daisy.run_blockwise([task])

    if ret:
        logger.info("Ran all blocks successfully!")
    else:
        logger.error("Did not run all blocks successfully...")

    return out_array
