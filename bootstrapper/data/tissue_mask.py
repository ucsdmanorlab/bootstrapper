import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import click
import edt
import numpy as np
import zarr
from funlib.persistence import open_ds, prepare_ds
from scipy import ndimage
from skimage.morphology import convex_hull_image

HALO = 4  # the z closing is a dilation by 2 then an erosion by 2
RULE = (
    "raw > 0 per section; a section below half the median coverage takes the OR of its "
    "nearest valid neighbours; binary_closing along z with ones((5,1,1)) on the edge-padded "
    "stack, z-filled voxels kept only inside the section's own convex hull; per section "
    "closing with the disk x^2 + y^2 <= {}^2 (two exact distance transforms), then fill holes"
)


def tissue_counts(raw, cz, z0):
    return np.count_nonzero(zarr.open(raw, "r")[z0 : z0 + cz], axis=(1, 2))


def neighbours(src, z, bad):
    lo = next((k for k in range(z - 1, -1, -1) if k not in bad), None)
    hi = next((k for k in range(z + 1, src.shape[0]) if k not in bad), None)
    return np.logical_or.reduce([src[k] > 0 for k in (lo, hi) if k is not None])


def hull(m):
    return convex_hull_image(m) | m if m.any() else m


def envelope(m, r):
    pad = 2 * r + 2
    grown = edt.edt(~np.pad(m, pad)) <= r
    closed = (edt.edt(grown) > r)[pad:-pad, pad:-pad]
    return ndimage.binary_fill_holes(closed | m)


def mask_block(raw, out, bad, radius, z0):
    src, dst = zarr.open(raw, "r"), zarr.open(out, "r+")
    z1 = min(z0 + dst.chunks[0], src.shape[0])
    r0, r1 = max(z0 - HALO, 0), min(z1 + HALO, src.shape[0])
    m = src[r0:r1] > 0
    for z in bad & set(range(r0, r1)):
        m[z - r0] = neighbours(src, z, bad)
    m = np.pad(m, ((2, 2), (0, 0), (0, 0)), mode="edge")
    closed = ndimage.binary_closing(m, np.ones((5, 1, 1), bool))
    own = slice(z0 - r0 + 2, z1 - r0 + 2)
    tissue = np.stack(
        [envelope(s | (c & hull(s)), radius) for s, c in zip(m[own], closed[own])]
    )
    dst[z0:z1] = tissue
    return np.count_nonzero(tissue)


@click.command()
@click.argument("raw", type=click.Path(exists=True))
@click.argument("out", type=click.Path())
@click.option(
    "--radius",
    "-r",
    type=int,
    default=24,
    show_default=True,
    help="Disk radius in voxels: 24 at 4 nm (about 100 nm), 12 at 8 nm",
)
@click.option(
    "--num-workers",
    "-w",
    type=int,
    default=8,
    show_default=True,
    help="Each worker needs about 80 bytes per voxel of one section: 3.5 GB at 4926 x 9750",
)
def tissue_mask(raw, out, radius, num_workers):
    """Tissue mask of a raw array, with no holes.

    Per section the filled envelope of raw > 0. Folds and black sections stay inside
    the mask, so bs refine mask keeps their labels.

    \b
    Example: bs utils tissue-mask vol.zarr/raw/s1 vol.zarr/masks/tissue/s1
    """
    if os.path.exists(out):
        raise click.ClickException(
            f"{out} already exists. Remove it or give another OUT."
        )
    ds = open_ds(raw)
    nz, cz = ds.shape[0], ds.chunk_shape[0]
    starts = range(0, nz, cz)
    with ProcessPoolExecutor(num_workers) as pool:
        counts = np.concatenate(list(pool.map(partial(tissue_counts, raw, cz), starts)))
        median = np.median(counts)
        bad = [int(z) for z in np.flatnonzero(counts < median / 2)]
        coverage = median / np.prod(ds.shape[1:])
        print(
            f"pass 1 of 2: {nz} sections, median tissue coverage {coverage:.0%}; "
            f"sections below half of it: {bad or 'none'}",
            flush=True,
        )
        prepare_ds(
            out,
            shape=ds.shape,
            offset=ds.roi.offset,
            voxel_size=ds.voxel_size,
            axis_names=ds.axis_names,
            units=ds.units,
            dtype=np.uint8,
            chunk_shape=ds.chunk_shape,
            custom_metadata={"rule": RULE.format(radius), "filled_sections": bad},
        )
        print(
            f"pass 2 of 2: {len(starts)} blocks of {cz} sections, {num_workers} workers",
            flush=True,
        )
        total = sum(pool.map(partial(mask_block, raw, out, set(bad), radius), starts))
    print(
        f"{out}: {total - counts.sum():,} voxels added to raw > 0; "
        f"sections filled from neighbours: {bad or 'none'}"
    )
