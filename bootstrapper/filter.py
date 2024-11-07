import click
import yaml
import json
import os
from ast import literal_eval
import glob
from pprint import pprint


DEFAULTS = {
    'dust_filter': 200,
    'remove_outliers': False,
    'remove_z_fragments': 4,
    'overlap_filter': 0.0,
    'exclude_ids': None,
    'erode_out_mask': False,
}


def get_best_seg_from_eval(eval_file):
    with open(eval_file, 'r') as f:
        results = json.load(f)

    # determine gt or self eval results
    test_result = list(results.values())[0]
    if "metrics" in test_result:
        if "voi" in test_result["metrics"]:
            metric = "voi_sum"
        elif "skel" in test_result["metrics"]:
            metric = "nerl"
        else:
            raise ValueError("Neither voi or skel results found in eval file")
    else:
        metric = "nonzero_ratio"

    # sort results by metric and return best seg
    if metric == "voi_sum":
        best_seg = sorted(results.items(), key=lambda x: x[1]["metrics"]["voi"]["voi_merge"] + x[1]["metrics"]["voi"]["voi_split"])[0][0]
    elif metric == "nerl":
        best_seg = sorted(results.items(), key=lambda x: x[1]["metrics"]["skel"]["nerl"], reverse=True)[0][0]
    elif metric == "nonzero_ratio":
        best_seg = sorted(results.items(), key=lambda x: x[1]["error_mask"]["nonzero_ratio"], reverse=True)[0][0]

    print(f"Best seg: {best_seg}")
    pprint(results[best_seg])

    return best_seg


def get_filter_config(yaml_file, **kwargs):
    # load config
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    for key, value in kwargs.items():
        if key != "param" and value is not None:
            config[key] = value

    # must contain eval results, or seg datasets
    out_seg_ds = config['out_seg_dataset']
    out_mask_ds = config['out_mask_dataset']
    in_error_mask_ds = config.get('in_error_mask_dataset', None)
    roi_offset = config.get('roi_offset', None)
    roi_shape = config.get('roi_shape', None)
    block_shape = config.get('block_shape', None)
    block_context = config.get('context', None)
    num_workers = config.get('num_workers', 20)

    if roi_offset is not None:
        roi_offset = literal_eval(roi_offset)
    if roi_shape is not None:
        roi_shape = literal_eval(roi_shape)
    if block_shape is not None and block_shape != "roi":
        block_shape = literal_eval(block_shape)
    if block_context is not None:
        block_context = literal_eval(block_context)
    
    # param override
    params = DEFAULTS.copy()
    if len(kwargs["param"]) > 0:
        for param in kwargs["param"]:
            p, v = param.split("=")
            params[p] = literal_eval(v)

    # if eval, get best seg from results
    in_seg_datasets = []
    if "eval_dir" in config:
        # get eval result files
        eval_files = glob.glob(os.path.join(config["eval_dir"], "*.json"))
        for eval_file in eval_files:
            in_seg_datasets.append(get_best_seg_from_eval(eval_file))
    elif "seg_container" in config and "seg_datasets_prefix" in config:
        seg_datasets = [
            x for x in glob.glob(os.path.join(config["seg_container"], config["seg_datasets_prefix"], "*", "*"))
            if os.path.isdir(x) and os.path.exists(os.path.join(x, ".zarray"))
        ]
        in_seg_datasets.extend(seg_datasets)
    elif "seg_datasets" in config:
        for x in config["seg_datasets"]:
            if os.path.exists(x) and os.path.exists(os.path.join(x, ".zarray")):
                in_seg_datasets.append(x)
            else:
                raise ValueError(f"Invalid seg_dataset: {x}")
    else:
        raise ValueError("Must provide either eval_dir, seg_container and seg_dataset_prefix, or seg_datasets")

    # output
    configs = []
    for i, in_seg_ds in enumerate(in_seg_datasets):
        configs.append({
            'seg_dataset':in_seg_ds,
            'out_labels_dataset': os.path.join(out_seg_ds, f"{i}"),
            'out_mask_dataset': os.path.join(out_mask_ds, f"{i}"),
            'error_mask_dataset': in_error_mask_ds,
            'roi_offset': roi_offset,
            'roi_shape': roi_shape,
            'block_shape': block_shape,
            'context': block_context,
            'num_workers': num_workers,
        } | params)

    return configs


def run_filter(config_file, **kwargs):
    from bootstrapper.post.blockwise.filter_segmentation import filter_segmentation

    # load config
    configs = get_filter_config(config_file, **kwargs)
    for config in configs:
        filter_segmentation(**config)


@click.command()
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False, file_okay=True))
@click.option("--roi-offset", "-ro", type=str, help="Offset of ROI in world units (literal eval of str)")
@click.option("--roi-shape", "-rs", type=str, help="Shape of ROI in world units (literal eval of str)")
@click.option("--num-workers","-n", type=int, help="Number of workers, for blockwise segmentation")
@click.option("--block-shape","-bs", type=str, help="Block shape, for blockwise segmentation (literal eval of str or 'roi')")
@click.option("--block-context","-bc", type=str, help="Block context, for blockwise segmentation (literal eval of str)")
@click.option("--param", "-p", multiple=True, help="Method specific parameters to override in config (e.g. -p 'remove_z_fragments=5')")
def filter(config_file, **kwargs):
    """Filter segmentations based on config file."""
    run_filter(config_file, **kwargs)
