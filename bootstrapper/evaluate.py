import click
import yaml
import glob
import os
import logging
import zarr
import json
from pprint import pprint

from funlib.persistence import open_ds

from .eval.compute_errors import compute_errors, compute_stats

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.group()
def evaluate():
    """
    Evaluate segmentations.
    
    Can evaluate segmentations against ground-truth labels, ground-truth skeletons or
    self-evaluate segmentations against predictions.
    """
    pass

@evaluate.command()
@click.argument("config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--out_result", type=click.Path())
def self(config_file, out_result):
    """
    Self-evaluate segmentations against predictions.

    Target predictions are computed from segmentations and compared to the model's predictions.
    """

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

   
    seg_container = config["seg_file"]
    seg_datasets_prefix = config["seg_datasets"]
    eval_dir = os.path.join(seg_container, config["out_dir"])

    mask_file = config.get("mask_file", None)
    mask_dataset = config.get("mask_dataset", None)
    mask_array = None if mask_file is None or mask_dataset is None else os.path.join(mask_file, mask_dataset)

    pred_dataset = config["self"]["pred_dataset"]
    thresholds = tuple(config["self"]["thresholds"])

    # search for every zarr array under seg_container/seg_datasets

    seg_datasets = []
    for ds in glob.glob(f"{seg_container}/{seg_datasets_prefix}/*/*/.zarray"):
        seg_datasets.append(os.path.dirname(ds))
    
    seg_stats = {}

    for seg_ds in seg_datasets:
        logger.info(f"Evaluating {seg_ds}")
        seg_name = seg_ds.split(f"{seg_datasets_prefix}/")[1]

        out_map_dataset = os.path.join(config["self"]["out_map_dataset"], seg_name)
        out_mask_dataset = os.path.join(config["self"]["out_mask_dataset"], seg_name)

        compute_errors(
            seg_ds,
            pred_dataset,
            mask_array,
            out_map_dataset,
            out_mask_dataset,
            thresholds=thresholds,
            return_arrays=False,
        )

        stats = {}
        stats["seg_ds"] = seg_ds
        stats["pred_ds"] = pred_dataset
        stats["mask_ds"] = mask_array
        stats["map_ds"] = out_map_dataset
        stats["mask_ds"] = out_mask_dataset
        stats["thresholds"] = thresholds
        pred_choice = "lsds" if "lsds_error_map" in out_map_dataset else "affs"

        stats[f"{pred_choice}_error_map"] = compute_stats(
            open_ds(out_map_dataset, mode="r")[:]
        )
        stats[f"{pred_choice}_error_mask"] = compute_stats(
            open_ds(out_mask_dataset, mode="r")[:]
        )

        logger.info(f"Stats for {seg_ds}: {pprint(stats)}")
        seg_stats[seg_ds] = stats

    # save stats
    if out_result is None:
        out_result = os.path.join(eval_dir, "self_eval_stats.json")

    logger.info(f"Saving stats to {out_result}")
    with open(out_result, "w") as f:
        json.dump(seg_stats, f, indent=4)