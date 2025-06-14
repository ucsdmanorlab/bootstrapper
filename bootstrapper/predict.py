import click
import os
import toml
import json
import daisy
import logging
import subprocess
from pprint import pprint

from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, prepare_ds

from .configs import download_checkpoints, MODEL_URLS
from .styles import cli_echo, cli_confirm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_blockwise(config):

    logger.info(
        f"Starting block-wise processing with predict_worker: {config['worker']}..."
    )

    # process block-wise
    task = daisy.Task(
        "PredictBlockwiseTask",
        config["total_roi"],
        config["read_roi"],
        config["write_roi"],
        process_function=lambda: call_predict(config),
        check_function=None,
        num_workers=config["num_workers"],
        read_write_conflict=False,
        max_retries=5,
        fit="overhang",
    )

    done = daisy.run_blockwise([task])
    if not done:
        raise RuntimeError("At least one block failed!")
    else:
        logger.info("All blocks finished successfully!")


def call_predict(config):
    worker_id = daisy.Context.from_env()["worker_id"]
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{int(worker_id) % config['num_gpus']}"
    subprocess.run(["python", config["worker"], *config["args"]])


def get_pred_config(config_file, setup_id, **kwargs):
    # load config
    with open(config_file, "r") as f:
        config = toml.load(f)[setup_id]

    # override config values with provided kwargs
    for key, value in kwargs.items():
        if value is not None:  # and len(value) > 0:
            config[key] = value

    setup_dir = config["setup_dir"]
    checkpoint = config["checkpoint"]
    input_datasets = config["input_datasets"]
    output_datasets_prefix = config["output_datasets_prefix"]
    chain_str = config.get("chain_str", "")
    num_workers = config.get("num_workers", 1)
    num_gpus = config.get("num_gpus", 1)
    roi_offset = config.get("roi_offset", None)
    roi_shape = config.get("roi_shape", None)

    # check if checkpoint exists
    if not os.path.exists(checkpoint):
        model_name = os.path.basename(setup_dir)
        if model_name in MODEL_URLS:
            cli_echo(f"{checkpoint} not found in {setup_dir}", "predict")

            download = cli_confirm(
                f"Enter whether to download pretrained checkpoints for {model_name}?", 
                "predict",
                default=True,
            )

            if download:
                download_checkpoints(model_name, setup_dir)
            else:
                raise ValueError(
                    f"Please either download checkpoints or train from scratch"
                )
        else:
            raise ValueError(f"Checkpoint {checkpoint} does not exist!")

    # try reading all input datasets, get voxel size
    in_channels_sum = 0
    for in_ds_path in input_datasets[::-1]:
        in_ds = open_ds(in_ds_path, "r")
        in_channels_sum += (
            in_ds.shape[0] if len(in_ds.shape) > 3 else 1
        )  # channels first, 3D
    in_ds = open_ds(input_datasets[0], "r")
    voxel_size = in_ds.voxel_size

    # load net config
    net_config_file = os.path.join(setup_dir, "net_config.json")
    with open(net_config_file, "r") as f:
        net_config = json.load(f)

    # validate number of input datasets
    assert len(input_datasets) == len(
        net_config["inputs"]
    ), f"number of input datasets ({len(input_datasets)}) does not match number of network inputs ({net_config['inputs']})"

    # get input, output shapes
    shape_increase = net_config["shape_increase"]
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]
    output_shape = [x + y for x, y in zip(shape_increase, net_config["output_shape"])]

    # add z-dimension for 2D networks
    if len(input_shape) == 2:
        input_shape = [
            net_config["in_channels"],
            *input_shape,
        ]  # support for "3ch" models
        output_shape = [1, *output_shape]

    # get block input and output ROIs
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = (input_size - output_size) / 2
    read_roi = Roi((0,) * len(input_size), input_size) - context
    write_roi = Roi((0,) * len(output_size), output_size)

    # get total roi
    if roi_offset is not None:
        input_roi = Roi(roi_offset, roi_shape).grow(context, context)
        output_roi = Roi(roi_offset, roi_shape)
    else:
        input_roi = in_ds.roi.grow(context, context)
        output_roi = in_ds.roi

    # get output dataset names and prepare output datasets if using daisy
    output_datasets = []
    iteration = checkpoint.split("_")[-1]
    for output_name, val in net_config["outputs"].items():
        output_dims = val["dims"]
        output_dtype = val["dtype"]

        out_ds = (
            f"{iteration}/{output_name}"
            if chain_str == ""
            else f"{iteration}--from--{chain_str}/{output_name}"
        )

        output_dataset = os.path.join(output_datasets_prefix, out_ds)
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

        # prepare output dataset
        prepare_ds(
            store=output_dataset,
            shape=(output_dims, *(output_roi.shape / voxel_size)),
            offset=output_roi.offset,
            voxel_size=voxel_size,
            axis_names=output_axes,
            units=in_ds.units,
            chunk_shape=(output_dims, *output_shape),
            dtype=output_dtype,
        )

    # get args
    worker = os.path.join(setup_dir, "predict.py")
    args = ["-c", checkpoint]
    for in_ds in input_datasets:
        args.extend(["-i", in_ds])
    for out_ds in output_datasets:
        args.extend(["-o", out_ds])

    # daisy for distributed prediction
    if num_gpus > 1:
        args.extend(["-d"])

        return {
            "total_roi": input_roi,
            "read_roi": read_roi,
            "write_roi": write_roi,
            "num_workers": num_workers,
            "num_gpus": num_gpus,
            "worker": worker,
            "args": args,
        }

    else:
        args.extend(["-n", str(num_workers)])
        args.extend(["-ro", " ".join(map(str, output_roi.offset))])
        args.extend(["-rs", " ".join(map(str, output_roi.shape))])

        return {
            "worker": worker,
            "args": args,
            "num_gpus": num_gpus,
            "num_workers": num_workers,
        }


def run_prediction(config_file, setup_ids=None, **kwargs):

    with open(config_file, "r") as f:
        all_setup_ids = list(toml.load(f).keys())

    valid_setups = {
        **{s.split("-")[0]: s for s in all_setup_ids},
        **{s.split("-")[-1]: s for s in all_setup_ids},
        **{s: s for s in all_setup_ids},
    }

    setups = sorted(setup_ids.strip().split()) if setup_ids else all_setup_ids

    for s_id in setups:
        if s_id not in valid_setups:
            raise ValueError(f"Setup ID {s_id} not found in {all_setup_ids}")

        config = get_pred_config(config_file, valid_setups[s_id], **kwargs)
        pprint(config)

        if config["num_gpus"] > 1:
            predict_blockwise(config)
        else:
            subprocess.run(["python", config["worker"], *config["args"]])


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option(
    "--setup-id", "-s", type=str, help="Setup ID(s) to run prediction for. 01, 02, etc."
)
@click.option(
    "--roi-offset",
    "-ro",
    type=str,
    help="Offset of ROI in world units (space separated integers)",
)
@click.option(
    "--roi-shape",
    "-rs",
    type=str,
    help="Shape of ROI in world units (space separated integers)",
)
@click.option("--num-workers", "-nw", type=int, help="Number of workers")
@click.option("--num-gpus", "-ng", type=int, help="Number of GPUs to use")
def predict(config_file, setup_id, **kwargs):
    """Run prediction for a setup or all setups in a prediction config file."""
    run_prediction(config_file, setup_id, **kwargs)


if __name__ == "__main__":
    predict()
