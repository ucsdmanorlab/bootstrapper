from functools import partial

import click
import numpy as np
import daisy
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, prepare_ds, Array
import fastremap


@click.group()
def refine():
    """Refine segmented volumes: size/outlier/z filtering, id remap, morphology."""
    pass


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _default_out(in_array, suffix):
    head, sep, tail = in_array.partition(".zarr/")
    if not sep or ".zarr/" in tail:
        raise click.ClickException(
            f"cannot derive an out_array from {in_array!r}; pass --out_array"
        )
    return f"{head}.zarr/{tail}_{suffix}"


def _prepare_like(in_ds, out_array):
    return prepare_ds(
        out_array,
        shape=in_ds.shape,
        offset=in_ds.roi.offset,
        voxel_size=in_ds.voxel_size,
        axis_names=in_ds.axis_names,
        units=in_ds.units,
        dtype=in_ds.dtype,
        chunk_shape=in_ds.chunk_shape,
        mode="w",
    )


def _run_blockwise(name, in_ds, out_ds, process_block, num_workers,
                   block_xy=2048, context=0, xy_only=True):
    voxel_size = in_ds.voxel_size
    dims = in_ds.roi.dims
    chunk = Coordinate(out_ds.chunk_shape)
    cz, cy, cx = int(chunk[0]), int(chunk[-2]), int(chunk[-1])
    block_size = Coordinate(
        (cz, cy * max(1, round(block_xy / cy)), cx * max(1, round(block_xy / cx)))
    )
    halo = Coordinate((0, context, context)) if xy_only else Coordinate((context,) * dims)

    write_block = daisy.Roi((0,) * dims, block_size * voxel_size)
    read_block = write_block.grow(halo * voxel_size, halo * voxel_size)
    total_roi = out_ds.roi.grow(halo * voxel_size, halo * voxel_size)

    task = daisy.Task(
        name,
        total_roi,
        read_block,
        write_block,
        process_function=process_block,
        read_write_conflict=False,
        num_workers=num_workers,
        max_retries=2,
        fit="shrink",
    )
    if not daisy.run_blockwise([task]):
        raise click.ClickException(f"{name}: some blocks failed (see daisy_logs/)")


def _scan_tiles(in_ds, tile=4096):
    voxel_size = in_ds.voxel_size
    offset = in_ds.roi.offset
    nz, ny, nx = in_ds.shape
    cz = int(in_ds.chunk_shape[0])
    chunk_xy = int(in_ds.chunk_shape[-1])
    tile = chunk_xy * max(1, round(tile / chunk_xy))
    for iz in range(0, nz, cz):
        sz = min(cz, nz - iz)
        for iy in range(0, ny, tile):
            for ix in range(0, nx, tile):
                sy, sx = min(tile, ny - iy), min(tile, nx - ix)
                roi = Roi(
                    offset + Coordinate(iz, iy, ix) * voxel_size,
                    Coordinate(sz, sy, sx) * voxel_size,
                )
                yield in_ds.to_ndarray(roi), iz


def _global_sizes(in_ds):
    """Total voxel count per nonzero label id, accumulated over the volume."""
    us, cs = [], []
    for data, _ in _scan_tiles(in_ds):
        u, c = fastremap.unique(data, return_counts=True)
        us.append(u)
        cs.append(c)
    uniq, inv = np.unique(np.concatenate(us), return_inverse=True)
    sizes = np.bincount(inv, weights=np.concatenate(cs).astype(np.float64)).astype(np.int64)
    fg = uniq != 0
    return uniq[fg], sizes[fg]


def _mask_block(in_ds, out_ds, remove_ids, block):
    data = in_ds.to_ndarray(block.write_roi)
    if remove_ids.size:
        fastremap.mask(data, remove_ids, in_place=True)  # listed ids -> 0
    out_ds[block.write_roi] = data
    return 0


def _finish_filter(in_ds, in_array, out_array, remove_ids, num_workers,
                   dry_run, suffix, name):
    if dry_run:
        print("dry run; nothing written")
        return
    out_array = out_array or _default_out(in_array, suffix)
    print(f"Writing to {out_array}")
    out_ds = _prepare_like(in_ds, out_array)
    _run_blockwise(name, in_ds, out_ds,
                   partial(_mask_block, in_ds, out_ds, remove_ids), num_workers)


# ---------------------------------------------------------------------------
# outlier_filter
# ---------------------------------------------------------------------------


@refine.command("outlier_filter")
@click.option("--in_array", "-i", type=click.Path(exists=True), required=True)
@click.option("--out_array", "-o", type=click.Path())
@click.option("--num_std", "-n", type=float, default=3.0,
              help="Cut objects further than num_std * std from the mean object size")
@click.option("--min_size", type=int, default=0,
              help="Objects smaller than this are excluded from the mean/std so debris "
              "does not skew the statistics")
@click.option("--num_workers", "-w", type=int, default=20)
@click.option("--dry_run", is_flag=True, default=False,
              help="Print the cutoffs and objects to remove without writing")
def outlier_filter(in_array, out_array, num_std, min_size, num_workers, dry_run):
    """Remove object-size outliers by a two-sided sigma cut, blockwise.

    Removes objects whose voxel count is more than num_std * std from the mean on
    either tail, large or small (statistics over objects >= min_size).
    """
    in_ds = open_ds(in_array)
    uniq, sizes = _global_sizes(in_ds)
    if uniq.size == 0:
        raise click.ClickException("no foreground objects in volume")

    stat_sizes = sizes[sizes >= min_size]
    if stat_sizes.size == 0:
        raise click.ClickException(f"no objects with size >= min_size ({min_size})")
    mean, std = float(stat_sizes.mean()), float(stat_sizes.std())
    lo, hi = mean - num_std * std, mean + num_std * std
    remove_ids = uniq[(sizes < lo) | (sizes > hi)]

    p50, p90, p99, p999 = np.percentile(stat_sizes, [50, 90, 99, 99.9])
    print(f"{uniq.size} objects; {stat_sizes.size} with size >= {min_size}")
    print(f"size p50={p50:.0f} p90={p90:.0f} p99={p99:.0f} p99.9={p999:.0f} "
          f"min={int(sizes.min())} max={int(sizes.max())}")
    print(f"mean={mean:.1f} std={std:.1f} | cut lo={lo:.1f} hi={hi:.1f} "
          f"-> removing {remove_ids.size} objects")

    _finish_filter(in_ds, in_array, out_array, remove_ids, num_workers,
                   dry_run, "outlier_filtered", "OutlierFilter")


# ---------------------------------------------------------------------------
# size_filter: keep objects with min_size <= size <= max_size
# ---------------------------------------------------------------------------


@refine.command("size_filter")
@click.option("--in_array", "-i", type=click.Path(exists=True), required=True)
@click.option("--out_array", "-o", type=click.Path())
@click.option("--min_size", type=int, default=0,
              help="Remove objects smaller than this many voxels (dust)")
@click.option("--max_size", type=int, default=None,
              help="Remove objects larger than this many voxels (0/unset = no cap)")
@click.option("--num_workers", "-w", type=int, default=20)
@click.option("--dry_run", is_flag=True, default=False)
def size_filter(in_array, out_array, min_size, max_size, num_workers, dry_run):
    """Remove objects outside a [min_size, max_size] voxel-count range, blockwise.

    Object identity is the existing global label id and its size is summed over
    the whole volume, so an object spanning blocks is judged by its total size,
    then the out-of-range ids are masked out block by block.
    """
    in_ds = open_ds(in_array)
    uniq, sizes = _global_sizes(in_ds)
    if uniq.size == 0:
        raise click.ClickException("no foreground objects in volume")

    remove = np.zeros(uniq.size, dtype=bool)
    if min_size > 0:
        remove |= sizes < min_size
    if max_size:
        remove |= sizes > max_size
    remove_ids = uniq[remove]

    print(f"{uniq.size} objects; sizes min={int(sizes.min())} max={int(sizes.max())} "
          f"median={int(np.median(sizes))}")
    print(f"range [{min_size}, {max_size}] -> removing {remove_ids.size} objects")

    _finish_filter(in_ds, in_array, out_array, remove_ids, num_workers,
                   dry_run, "size_filtered", "SizeFilter")


# ---------------------------------------------------------------------------
# z_filter: remove objects spanning <= min_z z-slices
# ---------------------------------------------------------------------------


@refine.command("z_filter")
@click.option("--in_array", "-i", type=click.Path(exists=True), required=True)
@click.option("--out_array", "-o", type=click.Path())
@click.option("--min_z", "-z", type=int, default=1,
              help="Remove objects whose z-extent is this many slices or fewer")
@click.option("--num_workers", "-w", type=int, default=20)
@click.option("--dry_run", is_flag=True, default=False)
def z_filter(in_array, out_array, min_z, num_workers, dry_run):
    """Remove thin objects that span few z-slices, blockwise.

    Computes each label's global z-extent (max z - min z + 1 over the whole
    volume) in a read-only pass, then masks out ids with extent <= min_z.
    """
    in_ds = open_ds(in_array)
    zmin, zmax = {}, {}
    for data, iz in _scan_tiles(in_ds):
        for z in range(data.shape[0]):
            gz = iz + z
            present = fastremap.unique(data[z])
            for lbl in present[present != 0].tolist():
                if lbl not in zmin:
                    zmin[lbl] = gz
                    zmax[lbl] = gz
                else:
                    if gz < zmin[lbl]:
                        zmin[lbl] = gz
                    if gz > zmax[lbl]:
                        zmax[lbl] = gz

    ids = np.fromiter(zmin.keys(), dtype=in_ds.dtype, count=len(zmin))
    spans = np.array([zmax[int(i)] - zmin[int(i)] + 1 for i in ids], dtype=np.int64)
    remove_ids = ids[spans <= min_z]
    print(f"{ids.size} objects; removing {remove_ids.size} with z-extent <= {min_z}")

    _finish_filter(in_ds, in_array, out_array, remove_ids, num_workers,
                   dry_run, "z_filtered", "ZFilter")


# ---------------------------------------------------------------------------
# remap: manual remove / merge of specific ids
# ---------------------------------------------------------------------------


def _remap_block(in_ds, out_ds, mapping, block):
    data = in_ds.to_ndarray(block.write_roi)
    fastremap.remap(data, mapping, preserve_missing_labels=True, in_place=True)
    out_ds[block.write_roi] = data
    return 0


@refine.command("remap")
@click.option("--in_array", "-i", type=click.Path(exists=True), required=True)
@click.option("--out_array", "-o", type=click.Path())
@click.option("--remove_ids", "-r", type=str, default=None,
              help="Comma-separated label ids to remove (set to 0)")
@click.option("--merge_ids", "-m", type=str, multiple=True,
              help="Comma-separated ids to merge into the first of the group "
              "(repeatable for multiple groups)")
@click.option("--num_workers", "-w", type=int, default=20)
def remap(in_array, out_array, remove_ids, merge_ids, num_workers):
    """Remove and/or merge specific label ids, blockwise."""
    remove = set()
    if remove_ids:
        remove = {int(x) for x in remove_ids.replace(" ", "").split(",")}
    merge = {}
    for group in merge_ids:
        ids = [int(x) for x in group.replace(" ", "").split(",")]
        for mid in ids:
            merge[mid] = ids[0]

    conflict = remove & set(merge)
    if conflict:
        raise click.ClickException(
            f"ids given to both --remove_ids and --merge_ids: {sorted(conflict)}"
        )
    mapping = {i: 0 for i in remove} | merge
    if not mapping:
        raise click.ClickException("nothing to do: pass --remove_ids and/or --merge_ids")
    print(f"remapping {len(mapping)} ids: {mapping}")

    in_ds = open_ds(in_array)
    out_array = out_array or _default_out(in_array, "remapped")
    print(f"Writing to {out_array}")
    out_ds = _prepare_like(in_ds, out_array)
    _run_blockwise("Remap", in_ds, out_ds,
                   partial(_remap_block, in_ds, out_ds, mapping), num_workers)


# ---------------------------------------------------------------------------
# morph: dilation / erosion / opening / closing / fill_holes
# ---------------------------------------------------------------------------

MORPH_OPS = ("dilate", "erode", "opening", "closing", "fill_holes")


def _fill_holes(data):
    import fastmorph

    two_d = data.ndim == 2
    vol = data[None] if two_d else data
    small, fwd = fastremap.renumber(vol, in_place=False)
    filled,_ = fastmorph.fill_holes_v2(small, fix_borders=two_d, merge_threshold=0.95)
    filled = fastremap.remap(filled, {v: k for k, v in fwd.items()},
                             preserve_missing_labels=True, in_place=True)
    return filled[0] if two_d else filled


def _apply_morph(data, op, iterations):
    import fastmorph

    if op == "dilate":
        return fastmorph.dilate(data, iterations=iterations)
    if op == "erode":
        return fastmorph.erode(data, iterations=iterations)
    if op == "opening": 
        return fastmorph.dilate(fastmorph.erode(data, iterations=iterations),
                                iterations=iterations)
    if op == "closing":
        return fastmorph.erode(fastmorph.dilate(data, iterations=iterations),
                               iterations=iterations)
    if op == "fill_holes":
        return _fill_holes(data)
    raise ValueError(op)


def _morph_block(in_ds, out_ds, op, iterations, xy, block):
    read_roi = block.read_roi.intersect(in_ds.roi)
    data = in_ds.to_ndarray(read_roi)

    if xy:
        out = np.empty_like(data)
        for z in range(data.shape[0]):
            out[z] = _apply_morph(data[z], op, iterations)
    else:
        out = _apply_morph(data, op, iterations)

    arr = Array(out, read_roi.offset, out_ds.voxel_size)
    out_ds[block.write_roi] = arr.to_ndarray(block.write_roi)
    return 0


@refine.command("morph")
@click.option("--in_array", "-i", type=click.Path(exists=True), required=True)
@click.option("--out_array", "-o", type=click.Path())
@click.option("--op", type=click.Choice(MORPH_OPS), required=True,
              help="Morphological operation")
@click.option("--iterations", "-n", type=int, default=1,
              help="Iterations for dilate/erode/opening/closing")
@click.option("--xy", is_flag=True, default=False,
              help="Apply per z-section (2D) instead of 3D")
@click.option("--context", "-c", type=int, default=64,
              help="Halo in voxels covering the op's reach (xy only with --xy, "
              "else xy and z); >= iterations for dilate/erode, "
              ">= 2*iterations for opening/closing, >= largest hole for fill_holes")
@click.option("--block_size", "-b", type=int, default=2048,
              help="XY write-block size in voxels (snapped to a chunk multiple; "
              "z tiles by the chunk)")
@click.option("--num_workers", "-w", type=int, default=20)
def morph(in_array, out_array, op, iterations, xy, context, block_size, num_workers):
    """Apply a morphological operation to a labelled volume, blockwise.

    Operations are label-preserving (multilabel). With --xy the operation runs on
    each z-section independently, otherwise in 3D over the block. The halo
    (--context) supplies neighbouring data so labels stay consistent across block
    seams; it is clipped at the true volume edge.
    """
    in_ds = open_ds(in_array)
    out_array = out_array or _default_out(in_array, op)
    print(f"Writing to {out_array}")
    out_ds = _prepare_like(in_ds, out_array)
    _run_blockwise(
        f"Morph-{op}",
        in_ds,
        out_ds,
        partial(_morph_block, in_ds, out_ds, op, iterations, xy),
        num_workers,
        block_xy=block_size,
        context=context,
        xy_only=xy,
    )
