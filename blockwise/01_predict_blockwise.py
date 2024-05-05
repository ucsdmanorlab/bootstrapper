import multiprocessing
multiprocessing.set_start_method('fork')

import yaml
import sys
import daisy
import json
import logging
import os
import re
import time
import subprocess

from pathlib import Path
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, prepare_ds

logging.getLogger().setLevel(logging.INFO)

NUM_GPUS = 4


def predict_blockwise(config: dict):

    print(config)

    setup_dir = config["setup_dir"]
    checkpoint = config["checkpoint"]
    raw_file = config["raw_file"]
    raw_datasets = config["raw_datasets"]
    out_file = config["out_file"]
    out_prefix = config["out_prefix"]
    num_workers = config["num_workers"]

    # from here on, all values are in world units (unless explicitly mentioned)
    # get ROI of source
    source = open_ds(raw_file, raw_datasets[0])
    logging.info(
        "Source dataset %s has shape %s, ROI %s, voxel size %s"
        % (raw_datasets[0], source.shape, source.roi, source.voxel_size)
    )
    
    # load net config
    with open(os.path.join(setup_dir, "config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "config.json")
        )
        net_config = json.load(f)

    outputs = net_config["outputs"]
    shape_increase = net_config["shape_increase"]
    input_shape = [x + y for x,y in zip(shape_increase,net_config["input_shape"])]
    output_shape = [x + y for x,y in zip(shape_increase,net_config["output_shape"])]

    # add z-dimension if 2D network
    if len(input_shape) == 2:
        input_shape = [3,*input_shape] if "3ch" in setup_dir else [1,*input_shape]
        output_shape = [1,*output_shape]

    # get chunk size and context
    net_input_size = Coordinate(input_shape) * source.voxel_size
    net_output_size = Coordinate(output_shape) * source.voxel_size
    context = (net_input_size - net_output_size) / 2

    logging.info(
        "Net input size is %s and net output size is %s"
        % (net_input_size, net_output_size)
    )
    
    # get total input and output ROIs
    if 'roi_offset' in config and 'roi_shape' in config:
        roi_offset = config['roi_offset']
        roi_shape = config['roi_shape']
    else:
        roi_offset = None
        roi_shape = None
    
    if roi_offset is not None:
        input_roi = Roi(roi_offset,roi_shape).grow(context,context)
        output_roi = Roi(roi_offset,roi_shape)
    else:
        input_roi = source.roi.grow(context, context)
        output_roi = source.roi

    # create read and write ROI
    ndims = source.roi.dims
    block_read_roi = Roi((0,) * ndims, net_input_size) - context
    block_write_roi = Roi((0,) * ndims, net_output_size)

    logging.info("Preparing output dataset ...")

    if out_prefix is None:
        out_prefix = ""

    # prepare output datasets
    iteration = checkpoint.split('_')[-1]
    out_dataset_names = []
    for output_name, val in outputs.items():
        out_dims = val["out_dims"]
        out_dtype = val["out_dtype"]
   
        if "2d_model" in setup_dir:
            assert len(raw_datasets) == 1
            out_dataset = os.path.join(out_prefix,raw_datasets[0],f"{output_name}_{iteration}")

        elif "2d_to_3d_model" in setup_dir:
            assert len(raw_datasets) == 2, f"{setup_dir} takes two inputs: stacked 2D LSDs and stacked 2D Affinities."
            out_dataset = f"{out_prefix}/{output_name}_{iteration}_from_{raw_datasets[0].split('_')[-1]}" 

        elif "3d_model" in setup_dir:
            assert len(raw_datasets) == 1
            out_dataset = f"{out_prefix}/{output_name}_{iteration}"

        # append to list to add to worker config
        out_dataset_names.append(out_dataset)

        prepare_ds(
            out_file,
            out_dataset,
            output_roi,
            source.voxel_size,
            out_dtype,
            write_size=block_write_roi.shape,
            num_channels=out_dims,
            compressor={"id": "blosc"},
            force_exact_write_size=True,
            delete=True,
        )

    # update config
    config["out_dataset_names"] = out_dataset_names
    predict_worker = os.path.abspath(os.path.join(setup_dir, "predict.py"))
    logging.info(f"Starting block-wise processing with predict_worker: {predict_worker}...")

    # process block-wise
    task = daisy.Task(
        "PredictBlockwiseTask",
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda: start_worker(config, predict_worker),
        check_function=None,
        num_workers=num_workers,
        read_write_conflict=False,
        max_retries=5,
        fit="overhang",
    )

    done = daisy.run_blockwise([task])

    if not done:
        raise RuntimeError("at least one block failed!")


def start_worker(config: dict, worker: str):
    config_file = Path(config["out_file"]) / "config.json"

    worker_id = daisy.Context.from_env()["worker_id"]
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{int(worker_id) % NUM_GPUS}"
   
    with open(config_file, "w") as f:
        json.dump(config, f)

    logging.info("Running block with config %s..." % config_file)

    subprocess.run(
        [
            "python",
            worker,
            str(config_file),
        ]
    )


if __name__ == "__main__":
    
    config_file = sys.argv[1]
    setup = sys.argv[2]

    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = yaml_config["predict"][setup]
    
    start = time.time()
    predict_blockwise(config)
    end = time.time()

    seconds = end - start
    logging.info(f'Total time to predict blockwise : {seconds} ')
