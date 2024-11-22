import click
import toml
import glob
import os
import logging
import json
from pprint import pprint

from funlib.persistence import open_ds


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_seg_datasets(seg_datasets_prefix):
    seg_datasets = []
    for ds in glob.glob(f"{seg_datasets_prefix}*/*/.zarray"):
        if "__vs__" not in ds:  # skip self errors
            seg_datasets.append(os.path.dirname(ds))
    return seg_datasets


def get_eval_config(config_file, mode, **kwargs):
    with open(config_file, "r") as f:
        config = toml.load(f)

    # Override config values with provided kwargs
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value

    if "out_result" not in config:
        config["out_result"] = config_file.replace("04_eval_",f"results_{mode}_").replace(".toml", ".json")

    return config


def run_gt_evaluation(config, seg_ds):
    from .eval.compute_metrics import compute_metrics

    gt_labels_dataset = config["gt"].get("labels_dataset")
    gt_skeletons_file = config["gt"].get("skeletons_file")
    mask_dataset = config.get("mask_dataset")

    if gt_labels_dataset is None and gt_skeletons_file is None:
        raise AssertionError("Either labels_dataset or skeletons_file must be provided")

    metrics = compute_metrics(
        seg_ds,
        gt_labels_dataset,
        gt_skeletons_file,
        mask_dataset,
    )

    stats = {
        "seg_ds": seg_ds,
        "labels_ds": gt_labels_dataset,
        "skeletons_file": gt_skeletons_file,
        "mask_ds": mask_dataset,
        "metrics": metrics,
    }

    return stats


def run_self_evaluation(config, seg_ds):
    from .eval.compute_errors import compute_errors, compute_stats

    pred_dataset = config["self"]["pred_dataset"]
    thresholds = tuple(config["self"]["thresholds"])
    params = config["self"].get("params", {})
    mask_dataset = config.get("mask_dataset")

    pred_name = os.path.basename(pred_dataset)
    out_map_dataset = seg_ds + f"__vs__{pred_name}"
    out_mask_dataset = seg_ds + f"__vs__{pred_name}"

    compute_errors(
        seg_ds,
        pred_dataset,
        mask_dataset,
        out_map_dataset,
        out_mask_dataset,
        thresholds=thresholds,
        return_arrays=False,
        **params,
    )

    stats = {
        "seg_ds": seg_ds,
        "pred_ds": pred_dataset,
        "mask_ds": mask_dataset,
        "map_ds": out_map_dataset,
        "mask_ds": out_mask_dataset,
        "thresholds": thresholds,
        "error_map": compute_stats(open_ds(out_map_dataset, mode="r")[:]),
        "error_mask": compute_stats(open_ds(out_mask_dataset, mode="r")[:]),
    }

    return stats


def run_evaluation(config_file, mode="pred", **kwargs):
    config = get_eval_config(config_file, mode, **kwargs)
    if "seg_datasets" in config:
        seg_datasets = [ds.rstrip("/") for ds in config["seg_datasets"]]
    else:
        seg_datasets = get_seg_datasets(config["seg_datasets_prefix"])
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

    out_result = kwargs.get("out_result") or config["out_result"]
    logger.info(f"Saving stats to {out_result}")
    with open(out_result, "w") as f:
        json.dump(seg_stats, f, indent=4)


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option("--gt", "-gt", is_flag=True, help="Evaluate only against ground-truth")
@click.option("--pred", "-p", is_flag=True, help="Evaluate only against predictions")
@click.option("--out_result", "-o", type=click.Path())
def evaluate(config_file, gt, pred, out_result=None):
    """
    Evaluate segmentations as specified in the config file.
    """

    eval_modes = []

    with open(config_file, "r") as f:
        config = toml.load(f)
        mode_configs = [config.get(mode, None) for mode in ["gt", "self"]]

    if any([gt, pred]):
        if gt:
            eval_modes.append("gt")
        if pred:
            eval_modes.append("pred")
    elif any(mode_configs):
        eval_modes = [mode for mode, mc in zip(["gt", "pred"], mode_configs) if mc]
    else:
        eval_modes = ["pred"]

    for mode in eval_modes:
        run_evaluation(config_file, mode, out_result=out_result)
