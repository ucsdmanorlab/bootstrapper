"""The run tail: the keys bs prepare derives for each step from the volumes and the keys
above. derive reads TOML and array metadata, never voxels. A key set above stays out of the
tail, and what follows from it follows the value set."""
import json
import os

import click
import zarr
from funlib.persistence import open_ds

from .catalog import MODELS, check_round
from .config import Step, ops_beside, runs, unset, with_tail
from .configs import MODEL_DIR, checkpoint_iteration, find_checkpoints
from .data.mask import training_mask
from .post.naming import mws_segmentation_name
from .segment import DEFAULTS as SEG_DEFAULTS

SHIPPED_ITERATION = 25000


def derive(volumes, steps, root):
    """The [run.<step>] table of each step, in step order. root holds the round's model
    setups and training masks."""
    chain = [_model(s) for s in steps if s.kind == "predict"]
    if chain and chain[0].reads_raw:
        check_round(chain)
    arrays = {n: {"raw": v.keys["raw"]} for n, v in volumes.items()}  # each volume's newest arrays by name
    models = {n: [] for n in volumes}  # each volume's <model>_<iteration> so far
    trained, segments, neighborhood, tails = {}, {}, None, []
    for step in steps:
        own, per = {}, {}  # derived keys of the step, and of each volume
        if step.kind == "train":
            labelled = {n: v.keys for n, v in volumes.items() if "labels" in v.keys}
            first = next(iter(labelled.values()))
            own = {"setup_dir": os.path.join(root, _model(step).name),
                   "voxel_size": first.get("voxel_size") or [int(x) for x in zarr.open(first["raw"], "r").attrs["voxel_size"]],
                   "samples": [_sample(n, keys, os.path.join(root, "training_mask.zarr", n)) for n, keys in labelled.items()]}
            run = _over(step, own, per)[0].keys
            last = os.path.join(run["setup_dir"], f"model_checkpoint_{run['max_iterations'] - 1}.ckpt")
            trained[step.name] = {"setup_dir": run["setup_dir"], "checkpoint": last}
        elif step.kind == "predict":
            model = _model(step)
            own = trained.get(model.name) or _shipped(model)
            per = {n: {"input_datasets": [arrays[n][x] for x in model.inputs],
                       "output_datasets_prefix": os.path.join(v.keys.get("outputs", _container(v.keys["raw"])), model.name),
                       "chain_str": "--from--".join(models[n]), **_box(v.keys)}
                   for n, v in volumes.items() if set(model.inputs) <= set(arrays[n])}
            run, configs = _over(step, own, per)
            for n, config in configs.items():
                if "checkpoint" not in config:
                    raise click.ClickException(
                        f"[{step.label}] needs setup_dir and checkpoint: no train step before it trains "
                        f"{model.name}, and {model.name} ships no checkpoint")
                iteration, chain_str = checkpoint_iteration(config["checkpoint"]), config.get("chain_str", "")
                level = f"{iteration}--from--{chain_str}" if chain_str else str(iteration)
                arrays[n] = {o: os.path.join(config["output_datasets_prefix"], level, o) for o in model.outputs}
                models[n].append(f"{model.name}_{iteration}")
            if "3d_affs" in model.outputs:
                neighborhood = _net_config(run.keys["setup_dir"], model)["outputs"]["3d_affs"]["neighborhood"]
        elif step.kind == "segment":
            own = {"params": {"aff_neighborhood": neighborhood}} if neighborhood else {}
            for n, v in volumes.items():
                if affs := (step.keys | step.volumes.get(n, {})).get("affs_dataset", arrays[n].get("3d_affs")):
                    level = os.path.dirname(affs)
                    per[n] = {"affs_dataset": affs, "fragments_dataset": os.path.join(level, "fragments_mws"),
                              "lut_dir": os.path.join(level, "luts_mws"),
                              "seg_dataset_prefix": os.path.join(level, "segmentations_mws"),
                              **_target(v.keys, "mask_dataset")}
            segments = _over(step, own, per)[1]
        elif step.kind == "refine":
            masked = "mask" in step.keys["ops"]
            per = {n: {"in_array": os.path.join(c["seg_dataset_prefix"], mws_segmentation_name(c | SEG_DEFAULTS["mws"] | c["params"])),
                       **({"ops": {"mask": _target(volumes[n].keys, "mask_array")}} if masked else {})}
                   for n, c in segments.items()}
        elif step.kind == "evaluate":
            per = {n: {"seg_datasets_prefix": c["seg_dataset_prefix"], **_target(volumes[n].keys, "mask_dataset"),
                       "gt": {"labels_dataset": volumes[n].keys["gt"]},
                       "pred": {"pred_dataset": c["affs_dataset"], "params": {"aff_neighborhood": c["params"]["aff_neighborhood"]}}}
                   for n, c in segments.items() if "gt" in volumes[n].keys}
        vols = {n: unset(keys, step.keys, step.volumes.get(n, {})) for n, keys in per.items()}
        tails.append(unset(own, step.keys) | {n: ops_beside(keys) for n, keys in vols.items() if keys})
    return tails


def _over(step, own, per):
    """The step over its derived keys, and each volume's config."""
    run = with_tail(step, Step(step.kind, step.name, own, per))
    return run, {n: config for n, config in runs(run) if n}


def _model(step):
    if step.name not in MODELS:
        raise click.ClickException(
            f"[{step.label}] names no model; bs prepare derives the steps named by a model, such as "
            f"[{step.kind}.2d_mtlsd]. Models: {', '.join(MODELS)}")
    return MODELS[step.name]


def _shipped(model):
    """The shipped setup and its newest checkpoint; none for a model you train."""
    if not model.shipped:
        return {}
    setup_dir = os.path.join(MODEL_DIR, model.name)
    found = find_checkpoints(setup_dir)
    return {"setup_dir": setup_dir,
            "checkpoint": found[max(found)] if found else os.path.join(setup_dir, f"model_checkpoint_{SHIPPED_ITERATION}")}


def _net_config(setup_dir, model):
    """The setup's net_config.json, or the shipped one before bs prepare copies the scripts."""
    path = os.path.join(setup_dir, "net_config.json")
    with open(path if os.path.exists(path) else os.path.join(MODEL_DIR, model.name, "net_config.json")) as f:
        return json.load(f)


def _sample(name, keys, mask_out):
    mask = training_mask(keys, mask_out)
    return {"volume": name, "raw": keys["raw"], "labels": keys["labels"], **({"mask": mask} if mask else {})}


def _box(keys):
    """The target mask's box cut to the raw, in nm."""
    if "target_mask" not in keys:
        return {}
    roi = open_ds(keys["target_mask"]).roi.intersect(open_ds(keys["raw"]).roi)
    return {"roi_offset": [int(x) for x in roi.offset], "roi_shape": [int(x) for x in roi.shape]}


def _target(keys, key):
    return {key: keys["target_mask"]} if "target_mask" in keys else {}


def _container(path):
    return path.rsplit(".zarr", 1)[0] + ".zarr"
