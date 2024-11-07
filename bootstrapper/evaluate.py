import click
import yaml
import glob
import os
import logging
import zarr
import json
from pprint import pprint

from funlib.persistence import open_ds


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_seg_datasets(seg_container, seg_datasets_prefix):
    seg_datasets = []
    for ds in glob.glob(f"{seg_container}/{seg_datasets_prefix}/*/*/.zarray"):
        seg_datasets.append(os.path.dirname(ds))
    return seg_datasets


def get_eval_config(yaml_file, **kwargs):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config values with provided kwargs
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value

    if "out_dir" not in config:
        config["out_dir"] = os.path.dirname(yaml_file)
    else:
        assert os.path.isdir(config["out_dir"]), f"Output directory {config['out_dir']} does not exist"
    
    return config


def run_gt_evaluation(config, seg_ds):
    from .eval.compute_metrics import compute_metrics

    gt_labels_dataset = config["gt"].get("labels_dataset")
    gt_skeletons_file = config["gt"].get("skeletons_file")
    
    if gt_labels_dataset is None and gt_skeletons_file is None:
        raise AssertionError("Either labels_dataset or skeletons_file must be provided")

    metrics = compute_metrics(
        seg_ds,
        gt_labels_dataset,
        gt_skeletons_file,
        config["mask_dataset"],
    )

    stats = {
        "seg_ds": seg_ds,
        "labels_ds": gt_labels_dataset,
        "skeletons_file": gt_skeletons_file,
        "mask_ds": config["mask_dataset"],
        "metrics": metrics
    }
    
    return stats

def run_self_evaluation(config, seg_ds):
    from .eval.compute_errors import compute_errors, compute_stats

    pred_dataset = config["self"]["pred_dataset"]
    thresholds = tuple(config["self"]["thresholds"])
    seg_name = seg_ds.split(f"{config['seg_datasets_prefix']}/")[1]
    
    out_map_dataset = os.path.join(config["self"]["out_map_dataset"], seg_name)
    out_mask_dataset = os.path.join(config["self"]["out_mask_dataset"], seg_name)

    compute_errors(
        seg_ds,
        pred_dataset,
        config["mask_dataset"],
        out_map_dataset,
        out_mask_dataset,
        thresholds=thresholds,
        return_arrays=False,
    )

    stats = {
        "seg_ds": seg_ds,
        "pred_ds": pred_dataset,
        "mask_ds": config["mask_dataset"],
        "map_ds": out_map_dataset,
        "mask_ds": out_mask_dataset,
        "thresholds": thresholds,
        "error_map": compute_stats(open_ds(out_map_dataset, mode="r")[:]),
        "error_mask": compute_stats(open_ds(out_mask_dataset, mode="r")[:])
    }
    
    return stats


def run_evaluation(yaml_file, mode="pred", **kwargs):
    config = get_eval_config(yaml_file, **kwargs)
    if "seg_datasets" in config:
        seg_datasets = config["seg_datasets"]
    else:
        seg_datasets = get_seg_datasets(config["seg_container"], config["seg_datasets_prefix"])
    seg_stats = {}

    for seg_ds in seg_datasets:
        print(f"Evaluating {seg_ds}")
        
        if mode == "pred":
            stats = run_self_evaluation(config, seg_ds)
        elif mode == "gt":
            stats = run_gt_evaluation(config, seg_ds)
            
        print(f"Stats for {seg_ds}:")
        pprint(stats)
        seg_stats[seg_ds] = stats

    out_result = kwargs.get("out_result") or os.path.join(config["out_dir"], f"{mode}_eval_stats.json")
    logger.info(f"Saving stats to {out_result}")
    with open(out_result, "w") as f:
        json.dump(seg_stats, f, indent=4)


@click.command()
@click.argument("yaml_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--gt","-gt", is_flag=True, help="Evaluate only against ground-truth")
@click.option("--pred","-p", is_flag=True, help="Evaluate only against predictions")
@click.option("--out_result", "-o", type=click.Path())
def evaluate(yaml_file, gt, pred, out_result):
    """
    Evaluate segmentations as specified in the config file.
    """
    modes = []
    if any([gt, pred]):
        if gt: modes.append("gt")
        if pred: modes.append("pred")
    else:
        modes = ["pred"]

    for mode in modes:
        run_evaluation(yaml_file, mode, out_result=out_result)
