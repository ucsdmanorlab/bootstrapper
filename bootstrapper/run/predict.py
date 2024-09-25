import click
import os
import yaml
import json
import daisy
import logging
import subprocess
from pprint import pprint

from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, prepare_ds

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_blockwise(
    worker,
    args,
    input_roi,
    block_read_roi,
    block_write_roi,
    num_workers,
    num_gpus,
):

    logger.info(f"Starting block-wise processing with predict_worker: {worker}...")

    # process block-wise
    task = daisy.Task(
        "PredictBlockwiseTask",
        input_roi,
        block_read_roi,
        block_write_roi,
        process_function=lambda: call_predict(worker, args, num_gpus),
        check_function=None,
        num_workers=num_workers,
        read_write_conflict=False,
        max_retries=5,
        fit="overhang",
    )

    done = daisy.run_blockwise([task])
    if not done:
        raise RuntimeError("At least one block failed!")
    else:
        logger.info("All blocks finished successfully!")


def call_predict(worker: str, args: list, num_gpus: int = 1):

    worker_id = daisy.Context.from_env()["worker_id"]
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{int(worker_id) % num_gpus}"

    subprocess.run(["python", worker, *args])


@click.command()
@click.argument(
    "yaml_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.argument("model_name", type=str)
@click.option(
    "--setup-dir",
    "-s",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to directory with model and configs",
)
@click.option(
    "--checkpoint",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to checkpoint file",
)
@click.option(
    "--input_datasets",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    multiple=True,
    help="Path to input dataset, can be specified multiple times",
)
@click.option(
    "--output_container", "-oc", type=click.Path(), help="Path to output zarr container"
)
@click.option(
    "--output_datasets_prefix", "-op", type=str, help="Prefix for output datasets names"
)
@click.option(
    "--roi_offset",
    "-ro",
    type=str,
    help="Offset of ROI in world units (space separated integers)",
)
@click.option(
    "--roi_shape",
    "-rs",
    type=str,
    help="Shape of ROI in world units (space separated integers)",
)
@click.option("--num_workers", "-nw", type=int, help="Number of workers")
@click.option("--num_gpus", "-gpu", type=int, help="Number of GPUs to use")
def predict(yaml_file, model_name, **kwargs):
    """
    Run prediction with the specified YAML configuration file and model name

    Overrides the values in the YAML file with the values provided on the command line.
    """

    # load config
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)[model_name]

    # override config values with command line arguments
    if any(kwargs.values()):
        for key, value in kwargs.items():
            if value is not None and len(value) > 0:
                config[key] = value

    logger.info(f"Using config: {pprint(config)}")

    setup_dir = config["setup_dir"]
    checkpoint = config["checkpoint"]
    input_datasets = config["input_datasets"]
    output_container = config["output_container"]
    output_datasets_prefix = config["output_datasets_prefix"]
    num_workers = config["num_workers"]

    # TODO, support device names
    if "num_gpus" in config:
        num_gpus = config["num_gpus"]
    else:
        num_gpus = 1

    # load net config
    net_config_file = os.path.join(setup_dir, "net_config.json")
    with open(net_config_file, "r") as f:
        net_config = json.load(f)

    # get net outputs, shapes
    outputs = net_config["outputs"]
    shape_increase = net_config["shape_increase"]
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]
    output_shape = [x + y for x, y in zip(shape_increase, net_config["output_shape"])]

    # add z-dimension if 2D network
    if len(input_shape) == 2:
        input_shape = [3, *input_shape] if "3ch" in setup_dir else [1, *input_shape]
        output_shape = [1, *output_shape]

    # get input datasets, voxel size
    in_ds = open_ds(input_datasets[0], "r")
    voxel_size = in_ds.voxel_size
    logger.info(
        f"Input dataset has shape {in_ds.shape}, ROI {in_ds.roi} and voxel size {voxel_size}"
    )

    # get block input and output ROIs
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = (input_size - output_size) / 2
    read_roi = Roi((0,) * len(input_size), input_size) - context
    write_roi = Roi((0,) * len(output_size), output_size)

    # get total input and output ROIs
    if "roi_offset" in config and "roi_shape" in config:
        roi_offset = config["roi_offset"]
        roi_shape = config["roi_shape"]
    else:
        roi_offset = None
        roi_shape = None

    if roi_offset is not None:
        input_roi = Roi(roi_offset, roi_shape).grow(context, context)
        output_roi = Roi(roi_offset, roi_shape)
    else:
        input_roi = in_ds.roi.grow(context, context)
        output_roi = in_ds.roi

    # get output dataset names and prepare output datasets
    output_datasets = []
    iteration = checkpoint.split("_")[-1]
    for output_name, val in outputs.items():
        out_dims = val["dims"]
        out_dtype = val["dtype"]

        # pred to pred
        if "_from_" in setup_dir:
            if "_from_2d_mtlsd" in setup_dir:
                assert (
                    len(input_datasets) == 2
                ), f"{setup_dir} takes two inputs: LSDs and Affinities."
            else:
                assert len(input_datasets) == 1

            out_ds = f"{output_datasets_prefix}/{output_name}_{iteration}_from_{input_datasets[0].split('_')[-1]}"

        # image to pred
        else:
            assert len(input_datasets) == 1
            out_ds = f"{output_datasets_prefix}/{output_name}_{iteration}"

        # append to list to add to worker config
        output_dataset = os.path.join(output_container, out_ds)
        output_datasets.append(output_dataset)
        output_axes = (
            [
                "c^",
            ]
            + in_ds.axis_names
            if "c^" not in in_ds.axis_names
            else in_ds.axis_names
        )

        logger.info(f"Preparing output dataset {output_dataset} with ROI {output_roi}")

        prepare_ds(
            store=output_dataset,
            shape=(out_dims, *(output_roi.shape / voxel_size)),
            offset=output_roi.offset,
            voxel_size=voxel_size,
            axis_names=output_axes,
            units=in_ds.units,
            chunk_shape=(out_dims, *output_shape),
            dtype=out_dtype,
        )

    # get command line arguments
    worker = os.path.join(setup_dir, "predict.py")
    args = ["-c", checkpoint]
    for in_ds in input_datasets:
        args.extend(["-i", in_ds])
    for out_ds in output_datasets:
        args.extend(["-o", out_ds])

    if "blockwise" in config and config["blockwise"] is True:
        args.extend(["-d"])
        predict_blockwise(
            worker,
            args,
            input_roi,
            read_roi,
            write_roi,
            num_workers,
            num_gpus,
        )
    else:
        args.extend(["-n", str(num_workers)])
        subprocess.run(["python", worker, *args])

if __name__ == "__main__":
    predict()