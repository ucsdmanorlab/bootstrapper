import gunpowder as gp
import json
import logging
import os
import click

from funlib.geometry import Coordinate, Roi
from funlib.persistence import open_ds

import torch
from model import AffsUNet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


@click.command()
@click.option(
    "--checkpoint",
    "-c",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to checkpoint file",
)
@click.option(
    "--input_datasets",
    "-i",
    required=True,
    multiple=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to input zarr datasets",
)
@click.option(
    "--output_datasets",
    "-o",
    required=True,
    multiple=True,
    type=click.Path(),
    help="Path to output zarr datasets",
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
@click.option("--num_workers", "-n", default=1, type=int, help="Number of workers")
@click.option(
    "--daisy", "-d", default=False, is_flag=True, help="Use daisy for parallelization"
)
def predict(
    checkpoint,
    input_datasets,
    output_datasets,
    roi_offset,
    roi_shape,
    num_workers,
    daisy,
):
    input_dataset = input_datasets[0]
    output_dataset = output_datasets[0]

    # load net config
    with open(os.path.join(setup_dir, "net_config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "net_config.json")
        )
        net_config = json.load(f)

    shape_increase = net_config["shape_increase"]
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]
    output_shape = [x + y for x, y in zip(shape_increase, net_config["output_shape"])]

    in_ds = open_ds(input_dataset, "r")
    voxel_size = in_ds.voxel_size
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size

    if not daisy:
        logging.info("Using Scan node with %d workers" % num_workers)
        if roi_offset is not None:
            roi_offset = Coordinate(roi_offset.split(" "))
            roi_shape = Coordinate(roi_shape.split(" "))
            roi = Roi(roi_offset, roi_shape).snap_to_grid(voxel_size, mode="grow")
        else:
            roi = in_ds.roi

    model = AffsUNet()
    
    checkpoint_path = checkpoint if os.path.exists(checkpoint) else f"{checkpoint}.ckpt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Neither {checkpoint} nor {checkpoint}.ckpt were found.")

    # Load the checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = state_dict.get('state_dict', state_dict.get('model_state_dict', state_dict))
    state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    input_lsds = gp.ArrayKey("INPUT_LSDS")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    chunk_request = gp.BatchRequest()
    chunk_request.add(input_lsds, input_size)
    chunk_request.add(pred_affs, output_size)

    source = gp.ArraySource(
        input_lsds,
        in_ds,
        interpolatable=True,
    )

    predict = gp.torch.Predict(
        model,
        inputs={
            "input_lsds": input_lsds,
        },
        outputs={
            0: pred_affs,
        },
        array_specs={pred_affs: gp.ArraySpec(roi=roi)} if not daisy else None,
    )

    scan = (
        gp.DaisyRequestBlocks(
            chunk_request,
            roi_map={
                input_lsds: "read_roi",
                pred_affs: "write_roi",
            },
        )
        if daisy
        else gp.Scan(chunk_request, num_workers=num_workers)
    )

    write = gp.ZarrWrite(
        dataset_names={pred_affs: output_dataset.split(".zarr")[-1]},
        store=output_dataset.split(".zarr")[0] + ".zarr",
    )

    pipeline = (
        source
        + gp.Normalize(input_lsds)
        + gp.Pad(input_lsds, None, mode="reflect")
        + gp.Unsqueeze([input_lsds])
        + predict
        + gp.Squeeze([pred_affs])
        + gp.IntensityScaleShift(pred_affs, 255, 0)
        + write
        + scan
    )

    predict_request = gp.BatchRequest()

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)


if __name__ == "__main__":
    predict()
