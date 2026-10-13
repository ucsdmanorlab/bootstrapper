import click
import toml
import glob
import os
import json

from funlib.persistence import open_ds

from .config import steps_of, runs


def get_seg_datasets(seg_datasets_prefix):
    seg_datasets = []
    for ds in sorted(glob.glob(f"{seg_datasets_prefix}*/*/.zarray")):
        if "__vs__" not in ds:  # skip pref errors
            seg_datasets.append(os.path.dirname(ds))
    return seg_datasets


def get_eval_config(config, config_file, mode, suffix=True, volume=None, **kwargs):
    config = dict(config)

    # Override config values with provided kwargs
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value

    if "out_result" not in config:
        config["out_result"] = config_file.replace("04_eval_","results_").replace(".toml", ".json")

    # each volume its own file; a mode suffix only when several modes run
    root, ext = os.path.splitext(config["out_result"])
    config["out_result"] = root + (f"_{volume}" if volume else "") + (f"_{mode}" if suffix else "") + ext

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


def run_pred_evaluation(config, seg_ds):
    from .eval.compute_errors import compute_errors, compute_stats

    pred_dataset = config["pred"]["pred_dataset"]
    thresholds = config["pred"].get("thresholds", config.get("thresholds", [0.1, 1.0]))
    params = config["pred"].get("params", {})
    mask_dataset = config.get("mask_dataset")

    pred_name = os.path.basename(pred_dataset)
    # out_dir: the segmentation may sit in a read-only container
    out_group = seg_ds + f"__vs__{pred_name}"
    if config.get("out_dir"):
        out_group = os.path.join(config["out_dir"], os.path.basename(seg_ds) + f"__vs__{pred_name}")
    out_map_dataset = os.path.join(out_group, "error_map")
    out_mask_dataset = os.path.join(out_group, "error_mask")

    compute_errors(
        seg_dataset=seg_ds,
        pred_dataset=pred_dataset,
        mask_dataset=mask_dataset,
        out_map_dataset=out_map_dataset,
        out_mask_dataset=out_mask_dataset,
        thresholds=thresholds,
        return_arrays=False,
        **params,
    )

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


def print_report(report):
    line = f"  {report['objects']} objects; foreground {report['foreground_of_roi']:.3f} of the ROI"
    if "outside_mask" in report:
        line += f", {report['foreground_of_mask']:.3f} of the mask; {report['outside_mask']} voxels outside the mask"
    print(line)
    if report["largest"]:
        first, *rest = (f"id {i} = {f:.3f}" for i, f in report["largest"])
        print("  largest: " + ", ".join([f"{first} of the ROI", *rest]))


def short_names(seg_datasets):
    """Names such as 'mws b-0.3/fill_holes': the method of segmentations_<method>, the name parts
    that differ between the segmentations, and the refine suffix. Full paths when names clash."""
    leaves = [os.path.basename(ds) for ds in seg_datasets]
    roots = [min((r for r in leaves if leaf.startswith(r + "_")), key=len, default=leaf) for leaf in leaves]
    shared = set.intersection(*(set(r.split("--")) for r in roots)) if any("--" in r for r in roots) else set()
    names = []
    for ds, leaf, root in zip(seg_datasets, leaves, roots):
        name = os.path.basename(os.path.dirname(ds)).removeprefix("segmentations_")
        parts = [p for p in root.split("--") if p not in shared]
        name += " " + "--".join(parts) if parts else ""
        names.append(name + ("/" + leaf[len(root) + 1:] if leaf != root else ""))
    return names if len(set(names)) == len(names) else list(seg_datasets)


def summary(stats):
    """The numbers of one segmentation that a user reads first."""
    if "metrics" in stats:
        parts = []
        if "voi" in stats["metrics"]:
            v = stats["metrics"]["voi"]
            parts.append(f"NVI split {v['nvi_split']:.3f} merge {v['nvi_merge']:.3f}  "
                         f"VOI split {v['voi_split']:.2f} merge {v['voi_merge']:.2f}")
        if "skel" in stats["metrics"]:
            s = stats["metrics"]["skel"]
            parts.append(f"nERL {s['nerl']:.3f}  {s['n_mergers']} merges {s['n_splits']} splits")
        return "  ".join(parts)
    if "error_mask" in stats:
        return f"error mask {stats['error_mask']['nonzero_ratio']:.3f} of the ROI"
    return f"{stats['report']['objects']} objects"


def run_evaluation(config, config_file, mode="pred", suffix=True, volume=None, **kwargs):
    """Report every segmentation, then evaluate it in mode gt or pred; mode report stops at the report.
    Returns one row per segmentation for the closing summary."""
    from .eval.compute_metrics import seg_report

    config = get_eval_config(config, config_file, mode, suffix, volume, **kwargs)
    if "seg_datasets" in config:
        seg_datasets = [ds.rstrip("/") for ds in config["seg_datasets"]]
    else:
        seg_datasets = get_seg_datasets(config["seg_datasets_prefix"])
        if not seg_datasets:
            raise click.ClickException(
                f"no segmentations under {config['seg_datasets_prefix']}; run the segment step first"
            )
    seg_stats = {}

    for seg_ds in seg_datasets:
        print(f"Evaluating {seg_ds}")
        report = seg_report(seg_ds, config.get("mask_dataset"))
        print_report(report)
        seg_stats[seg_ds] = {"report": report}
        if mode == "report":
            continue

        if mode == "pred":
            stats = run_pred_evaluation(config, seg_ds)
        elif mode == "gt":
            stats = run_gt_evaluation(config, seg_ds)

        seg_stats[seg_ds] |= stats

    out_result = config["out_result"]
    with open(out_result, "w") as f:
        json.dump(seg_stats, f, indent=4)
    names = short_names(seg_datasets)
    return [(volume or "", name, summary(seg_stats[ds]), f"({out_result})") for ds, name in zip(seg_datasets, names)]


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option("--step", type=str, help="The evaluate step to run when the file has several")
@click.option("--gt", "-gt", is_flag=True, help="Evaluate only against ground-truth")
@click.option("--pred", "-p", is_flag=True, help="Evaluate only against predictions")
@click.option("--out-result", "-o", type=click.Path())
def evaluate(config_file, step, gt, pred, out_result=None):
    """Evaluate segmentations as the [evaluate] step of config_file says."""
    steps = steps_of(config_file, "evaluate", step)
    rows = []
    for s in steps:
        flags = [m for m, on in (("gt", gt), ("pred", pred)) if on]
        for volume, config in runs(s):
            modes = flags or [m for m in ("gt", "pred") if m in config] or ["report"]
            for mode in modes:
                rows += run_evaluation(config, config_file, mode, len(modes) > 1, volume, out_result=out_result)
    widths = [max((len(row[i]) for row in rows), default=0) for i in range(3)]
    for row in rows:
        click.echo("  ".join([*(c.ljust(w) for c, w in zip(row, widths) if w), row[3]]))
