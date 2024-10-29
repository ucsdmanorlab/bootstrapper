import click
import logging
import yaml

from .blockwise.hglom.frags import frags, extract_fragments
from .blockwise.hglom.agglom import agglom, agglomerate
from .blockwise.hglom.luts import luts, find_segments
from .blockwise.hglom.extract import extract, extract_segmentations


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# class OrderedGroup(click.Group):
#     def list_commands(self, ctx):
#         # Return the commands in the desired order
#         return [
#             "frags",
#             "agglom",
#             "luts",
#             "extract"
#         ]

# @click.group(invoke_without_command=True, cls=OrderedGroup, chain=True)
# @click.argument("config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
# @click.pass_context
# def ws(ctx, config_file):
#     """
#     Hierarchical region agglomeration using waterz.

#     No subcommand runs the whole pipeline:
#     frags -> agglom -> luts -> extract
#     """
#     if ctx.invoked_subcommand is None:

#         with open(config_file, "r") as f:
#             config = yaml.safe_load(f)

#         waterz_pipeline(config)
#     pass

# ws.add_command(frags)
# ws.add_command(agglom)
# ws.add_command(luts)
# ws.add_command(extract)


def waterz_pipeline(config):
    extract_fragments(config)
    agglomerate(config)
    find_segments(config)
    extract_segmentations(config)


def simple_watershed(config):
    import os
    import numpy as np
    from funlib.persistence import open_ds, prepare_ds
    from funlib.geometry import Roi
    from .ws import watershed_from_affinities
    import waterz

    affs_ds = config["affs_dataset"]
    frags_ds = config["fragments_dataset"]
    seg_file = config["seg_file"]
    seg_ds_prefix = config["seg_dataset_prefix"]
    mask_ds = config.get("mask_dataset", None)
    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)

    # optional waterz params
    thresholds = config.get("thresholds", [0.5])
    fragments_in_xy = config.get("fragments_in_xy", True)
    min_seed_distance = config.get("min_seed_distance", 10)

    assert len(thresholds) == 1, "Only one threshold supported for simple watershed"

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
        mask = mask[roi].to_ndarray()
    else:
        mask = None

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
    )

    segmentation = next(generator)

    # write segmentation
    seg_ds_name = os.path.join(seg_file, seg_ds_prefix, "watershed", str(threshold))
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
