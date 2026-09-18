import click
import toml
import glob
import multiprocessing
import os
import logging
import json
from pprint import pprint

from funlib.persistence import open_ds


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_seg_datasets(seg_datasets_prefix):
    seg_datasets = []
    for ds in sorted(glob.glob(f"{seg_datasets_prefix}*/*/.zarray")):
        if "__vs__" not in ds:  # skip pref errors
            seg_datasets.append(os.path.dirname(ds))
    return seg_datasets


def get_eval_config(config_file, mode, suffix=True, **kwargs):
    with open(config_file, "r") as f:
        config = toml.load(f)

    # Override config values with provided kwargs
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value

    if "out_result" not in config:
        config["out_result"] = config_file.replace("04_eval_","results_").replace(".toml", ".json")

    # one file per mode when several modes run, so gt results do not overwrite
    # pred results; a single mode keeps the name it was given
    if suffix:
        root, ext = os.path.splitext(config["out_result"])
        config["out_result"] = f"{root}_{mode}{ext}"

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


def _compute_errors(kwargs):
    from .eval.compute_errors import compute_errors

    compute_errors(**kwargs)


def run_pred_evaluation(config, seg_ds):
    from .eval.compute_errors import compute_errors, compute_stats

    pred_dataset = config["pred"]["pred_dataset"]
    thresholds = config["pred"].get("thresholds", [0.1, 1.0])
    params = config["pred"].get("params", {})
    mask_dataset = config.get("mask_dataset")

    pred_name = os.path.basename(pred_dataset)
    # error maps land beside the segmentation unless out_dir says where; the
    # segmentation may live in a read-only container (a ground-truth labels array)
    out_group = seg_ds + f"__vs__{pred_name}"
    if config.get("out_dir"):
        out_group = os.path.join(config["out_dir"], os.path.basename(seg_ds) + f"__vs__{pred_name}")
    out_map_dataset = os.path.join(out_group, "error_map")
    out_mask_dataset = os.path.join(out_group, "error_mask")

    kwargs = dict(
        seg_dataset=seg_ds,
        pred_dataset=pred_dataset,
        mask_dataset=mask_dataset,
        out_map_dataset=out_map_dataset,
        out_mask_dataset=out_mask_dataset,
        thresholds=thresholds,
        return_arrays=False,
        num_workers=config.get("num_workers", 1),
        **params,
    )
    if kwargs["num_workers"] > 1:
        # gunpowder forks its scan workers; after one scan the parent holds thread
        # pools that deadlock the next fork, so every scan starts from a fresh process
        proc = multiprocessing.get_context("spawn").Process(target=_compute_errors, args=(kwargs,))
        proc.start()
        proc.join()
        if proc.exitcode:
            raise click.ClickException(f"error map for {seg_ds} failed (exit code {proc.exitcode})")
    else:
        compute_errors(**kwargs)

    stats = {
        "seg_ds": seg_ds,
        "pred_ds": pred_dataset,
        "mask_ds": mask_dataset,
        "error_map_ds": out_map_dataset,
        "error_mask_ds": out_mask_dataset,
        "thresholds": thresholds,
        "error_map": compute_stats(open_ds(out_map_dataset, mode="r")[:]),
        "error_mask": compute_stats(open_ds(out_mask_dataset, mode="r")[:]),
    }

    return stats


def run_evaluation(config_file, mode="pred", suffix=True, **kwargs):
    config = get_eval_config(config_file, mode, suffix, **kwargs)
    if "seg_datasets" in config:
        seg_datasets = [ds.rstrip("/") for ds in config["seg_datasets"]]
    else:
        seg_datasets = get_seg_datasets(config["seg_datasets_prefix"])
    seg_stats = {}

    for seg_ds in seg_datasets:
        print(f"Evaluating {seg_ds}")

        if mode == "pred":
            stats = run_pred_evaluation(config, seg_ds)
        elif mode == "gt":
            stats = run_gt_evaluation(config, seg_ds)

        print(f"Stats for {seg_ds}:")
        pprint(stats)
        seg_stats[seg_ds] = stats

    out_result = config["out_result"]
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
        mode_configs = [config.get(mode, None) for mode in ["gt", "pred"]]

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
        run_evaluation(config_file, mode, suffix=len(eval_modes) > 1, out_result=out_result)
