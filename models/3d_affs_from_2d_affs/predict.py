import json
import gunpowder as gp
import os
import sys
import logging
import zarr
from funlib.geometry import Coordinate

from model import AffsUNet

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def predict(config):
    checkpoint = config["checkpoint"]
    input_file = config["raw_file"]
    input_datasets = config["raw_datasets"] # [stacked_affs_x]
    out_file = config["out_file"]
    num_cache_workers = config["num_cache_workers"]

    # load net config
    with open(os.path.join(setup_dir, "net_config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "net_config.json")
        )
        net_config = json.load(f)

    out_dataset = config["out_dataset_names"][0]

    shape_increase = net_config["shape_increase"]
    input_shape = [x + y for x,y in zip(shape_increase,net_config["input_shape"])]
    output_shape = [x + y for x,y in zip(shape_increase,net_config["output_shape"])]
   
    voxel_size = Coordinate(zarr.open(input_file,"r")[input_datasets[0]].attrs["resolution"])
    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size
    context = (input_size - output_size) / 2
    
    model = AffsUNet()
    model.eval()

    input_affs = gp.ArrayKey('INPUT_AFFS')
    pred_affs = gp.ArrayKey('PRED_AFFS')

    chunk_request = gp.BatchRequest()
    chunk_request.add(input_affs, input_size)
    chunk_request.add(pred_affs, output_size)

    source = gp.ZarrSource(
                input_file,
            {
                input_affs: input_datasets[0]
            },
            {
                input_affs: gp.ArraySpec(interpolatable=True)
            })

    predict = gp.torch.Predict(
            model,
            checkpoint=checkpoint,
            inputs = {
                'input_affs': input_affs
            },
            outputs = {
                0: pred_affs,
            })

    scan = gp.DaisyRequestBlocks(
            chunk_request,
            roi_map={
                input_affs: 'read_roi',
                pred_affs: 'write_roi'
            },
            num_workers=num_cache_workers)

    write = gp.ZarrWrite(
            dataset_names={
                pred_affs: out_dataset,
            },
            store=out_file)

    pipeline = (
            source +
            gp.Normalize(input_affs) +
            gp.Pad(input_affs, None, mode="reflect") +
            gp.Unsqueeze([input_affs]) +
            predict +
            gp.Squeeze([pred_affs]) +
            gp.IntensityScaleShift(pred_affs,255,0) +
            write+
            scan)

    predict_request = gp.BatchRequest()

    with gp.build(pipeline):
        pipeline.request_batch(predict_request)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        run_config = json.load(f)

    predict(run_config)

