import re
import yaml
import os
import numpy as np
import zarr
import sys
import logging
from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds, prepare_ds
import daisy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def copy(
        block,
        in_ds,
        out_ds):

    logger.debug("Block: %s" % block)
    
    roi_2d = Roi(block.read_roi.offset[1:], block.read_roi.shape[1:])

    in_array = in_ds[roi_2d]
    logger.debug("Got data of shape %s" % str(in_array.shape))

    out_ds[block.write_roi] = np.expand_dims(in_array.to_ndarray(),axis=-3)

    logger.debug("Done.")


def get_2d_sections(config: dict):
    if config["sections"] is None:
        # find all sections
        sections = sorted([
            x for x in os.listdir(
                os.path.join(
                    config["raw_file"],
                    config["raw_datasets_prefix"][0])
                )
            if '.' not in x
        ])

        # natural sort
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        sections = sorted(sections, key=alphanum_key)
        stack_order = None

    else:
        # construct sections
        start, end, step, include, exclude, stack_order = config["sections"].values()

        sections = list(range(start,end,step))
        sections += include
        for x in exclude:
            sections.remove(x)
        sections.sort()

    if stack_order is None:
        stack_order = sections.copy()

    return sections, stack_order


def stack_datasets(
        zarr_container: str,
        input_datasets: list[str],
        out_ds_name: str,
        z_resolution: int = 50,
        out_ds_roi_offset = None, #list[int] = [0,0,0],
        num_workers: int = 120):

    in_ds = open_ds(zarr_container,input_datasets[0]) 

    # add number of sections to final volume shape
    shape = list(in_ds.shape)
    if len(shape) == 3:
        shape.insert(1,len(input_datasets))
        num_channels = shape[0]
    elif len(shape) == 2:
        shape.insert(0,len(input_datasets))
        num_channels = None

    # do blockwise only for large sections
    if shape[-1] >= 4096 or shape[-2] >= 4096:
        blockwise = True
    else:
        blockwise = False
    print(shape, f"doing blockwise={blockwise}")

    # get voxel sizes
    voxel_size = list(in_ds.voxel_size)
    voxel_size_3d = Coordinate([z_resolution,] + voxel_size)

    # get output roi
    if out_ds_roi_offset is not None:
        out_ds_roi_offset = Coordinate(out_ds_roi_offset)
    else:
        out_ds_roi_offset = Coordinate((0,0,0))

    total_3d_roi = Roi(out_ds_roi_offset, Coordinate(shape[-3:]) * voxel_size_3d)

    # preserve chunk shape as block shape
    chunk_shape_2d = list(in_ds.chunk_shape)[-2:]
    block_shape = Coordinate([len(input_datasets),] + chunk_shape_2d) * voxel_size_3d
    read_roi = write_roi = Roi((0,0,0), block_shape)
    print(chunk_shape_2d, voxel_size_3d, block_shape)

    # prepare output ds
    out_ds = prepare_ds(
        zarr_container,
        out_ds_name,
        total_roi=total_3d_roi,
        voxel_size=voxel_size_3d,
        dtype=np.uint8,
        num_channels=num_channels,
        write_size=write_roi.shape,
        force_exact_write_size=True,
        delete=True)

    for i, in_ds_name in enumerate(input_datasets):
        print(f"Copying {in_ds_name} to {out_ds_name}, blockwise={blockwise}")

        in_ds = open_ds(zarr_container, in_ds_name)
    
        total_roi = Roi(
                Coordinate(z_resolution*i,0,0) + Coordinate(out_ds_roi_offset),
                Coordinate([1,shape[-2],shape[-1]]) * voxel_size_3d
        )
        
        if not blockwise:
            out_ds[total_roi] = np.expand_dims(in_ds.to_ndarray(),axis=-3)

        if blockwise:
            task = daisy.Task(
                f"StackTask_{i}/{shape[-3]}",
                total_roi=total_roi,
                read_roi=read_roi,
                write_roi=write_roi,
                process_function=lambda b: copy(b,in_ds,out_ds),
                check_function=None,
                num_workers=num_workers,
                read_write_conflict=True,
                fit="shrink"
            )

            done = daisy.run_blockwise([task])

            if not done:
                raise RuntimeError("at least one block failed!")


if __name__ == "__main__":

    zarr_container = sys.argv[1]
    out_ds_name = sys.argv[2]
    z_resolution = int(sys.argv[3]) # nm / px
    input_datasets = sys.argv[4:]

    stack_datasets(
            zarr_container,
            input_datasets,
            out_ds_name,
            z_resolution)
