from model import Model
from funlib.geometry import Coordinate
import gunpowder as gp
import json
import logging
import os
import sys
import zarr

from funlib.persistence import open_ds

logging.basicConfig(level=logging.INFO)
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def predict(config):
    checkpoint = config["checkpoint"]
    raw_file = config["raw_file"]
    raw_dataset = config["raw_datasets"][0]
    out_file = config["out_file"]
    out_dataset_names = config["out_dataset_names"]
    num_cache_workers = config["num_cache_workers"]

    out_lsds_dataset = out_dataset_names[0]

    # load net config
    with open(os.path.join(setup_dir, "net_config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "net_config.json")
        )
        net_config = json.load(f)

    shape_increase = net_config["shape_increase"]
    input_shape = [
        1,
        *[x + y for x, y in zip(shape_increase, net_config["input_shape"])],
    ]
    output_shape = [
        1,
        *[x + y for x, y in zip(shape_increase, net_config["output_shape"])],
    ]

    voxel_size = Coordinate(zarr.open(raw_file, "r")[raw_dataset].attrs["voxel_size"])
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = (input_size - output_size) / 2

    model = Model(stack_infer=True)
    model.eval()

    raw = gp.ArrayKey("RAW")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

    chunk_request = gp.BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(pred_lsds, output_size)

    source = gp.ArraySource(
        raw, open_ds(os.path.join(raw_file, raw_dataset)), interpolatable=True
    )

    predict = gp.torch.Predict(
        model,
        checkpoint=checkpoint,
        inputs={"input": raw},
        outputs={
            0: pred_lsds,
        },
    )

    write = gp.ZarrWrite(
        dataset_names={
            pred_lsds: out_lsds_dataset,
        },
        store=out_file,
    )

    scan = gp.DaisyRequestBlocks(
        chunk_request,
        roi_map={
            raw: "read_roi",
            pred_lsds: "write_roi",
        },
        num_workers=num_cache_workers,
    )

    pipeline = (
        source
        + gp.Normalize(raw)
        + gp.Pad(raw, None, mode="reflect")
        + gp.IntensityScaleShift(raw, 2, -1)
        + gp.Unsqueeze([raw])
        + predict
        + gp.Squeeze([pred_lsds])
        + gp.Normalize(pred_lsds)
        + gp.IntensityScaleShift(pred_lsds, 255, 0)
        + write
        + scan
    )

    predict_request = gp.BatchRequest()

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        run_config = json.load(f)

    predict(run_config)
