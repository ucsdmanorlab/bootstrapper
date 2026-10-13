"""bs prepare: ask about the data, then write the round directory."""
import json
import os

import click
import toml
import zarr

from .catalog import check_round, default_chain, describe, parse_chain, valid_chains
from .configs import BS_DIR, EVAL_ERROR_THRESHOLDS, checkpoint_iteration, copy_model_scripts, find_checkpoints
from .data.convert import convert
from .data.mask import training_mask
from .post.naming import mws_segmentation_name
from .segment import DEFAULTS as SEG_DEFAULTS
from .styles import cli_echo, cli_prompt, cli_confirm

STYLE = "prepare"
SHIPPED_ITERATION = 25000


def ask_volume(round_dir):
    raw = os.path.abspath(cli_prompt("Raw image: zarr array, tif, or directory of 2D images", STYLE, type=click.Path(exists=True)))
    if not os.path.exists(os.path.join(raw, ".zarray")):
        name = cli_prompt("Volume name", STYLE, default=os.path.splitext(os.path.basename(raw))[0])
        voxel_size = cli_prompt("Voxel size in z y x (nm)", STYLE)
        container = os.path.join(round_dir, f"{name}.zarr")
        click.get_current_context().invoke(convert, in_path=raw, out_array=os.path.join(container, "raw"), voxel_size=voxel_size)
        raw = os.path.join(container, "raw")
    volume = {"raw": raw}
    container = raw.rsplit(".zarr", 1)[0] + ".zarr"
    name = os.path.basename(container)[: -len(".zarr")]
    attrs = zarr.open(raw, "r").attrs
    if "voxel_size" not in attrs:
        volume["voxel_size"] = [int(v) for v in cli_prompt(f"{raw} has no voxel_size; voxel size in z y x (nm)", STYLE).split()]
    if not os.access(container, os.W_OK):
        volume["outputs"] = os.path.join(round_dir, f"{name}.zarr")

    labels = cli_prompt("Labels array (enter to skip: the volume is only predicted)", STYLE, default="", show_default=False)
    if labels:
        volume["labels"] = os.path.abspath(labels)
        volume["training_mask"] = "everywhere" if cli_confirm("Is every object in this volume labelled?", STYLE, default=False) else "labels"
        volume["2d_labels"] = cli_confirm("Do the labels sit on single sections?", STYLE, default=labels_look_2d(volume["labels"]))
        if volume["training_mask"] == "labels" and not cli_confirm("Are the unlabelled neighbours of a labelled object other objects (yes), or background (no)?", STYLE, default=True):
            volume["mask_dilation"] = 10
    mask = cli_prompt("Mask array for the raw image (enter to skip)", STYLE, default="", show_default=False)
    if mask:
        volume["target_mask"] = os.path.abspath(mask)
    gt = cli_prompt("Ground truth labels for evaluation (enter to skip)", STYLE, default="", show_default=False)
    if gt:
        volume["gt"] = os.path.abspath(gt)
    return name, volume


def labels_look_2d(path):
    """Fewer than half of sixteen sampled sections hold a label."""
    arr = zarr.open(path, "r")
    zs = range(0, arr.shape[0], max(1, arr.shape[0] // 16))
    return sum(bool(arr[z].any()) for z in zs) < len(zs) / 2


def ask_models(volumes):
    default = " ".join(m.name for m in default_chain(any(v.get("2d_labels") for v in volumes.values())))
    chains = valid_chains()
    click.echo()
    while True:
        text = cli_prompt(f"Models (? lists all {len(chains)})", STYLE, default=default)
        if text.strip() == "?":
            for chain in chains:
                cli_echo(f"  {' '.join(m.short for m in chain):9s} {describe(chain)}", STYLE)
            cli_echo("  Type names or short forms, such as 2M 3Af2L.", STYLE)
            continue
        try:
            chain = parse_chain(text)
            check_round(chain)
        except click.ClickException as e:
            cli_echo(e.message, STYLE, "error")
            continue
        cli_echo(f"  {describe(chain)}", STYLE)
        return chain


def ask_gpus():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    granted = len([d for d in visible.split(",") if d.strip()]) if visible else 1
    return cli_prompt("GPUs for prediction", STYLE, type=int, default=max(1, granted))


def net_config(setup_dir):
    with open(os.path.join(setup_dir, "net_config.json")) as f:
        return json.load(f)


def write_round(round_dir, volumes, chain, num_gpus):
    trained = [v for v in volumes.values() if "labels" in v]
    if not trained:
        raise click.ClickException("no volume has labels; nothing to train")
    written = []

    def write(name, doc):
        path = os.path.join(round_dir, name)
        with open(path, "w") as f:
            toml.dump(doc, f)
        written.append(path)

    write("volumes.toml", {"volumes": volumes})
    voxel_size = trained[0].get("voxel_size") or [int(v) for v in zarr.open(trained[0]["raw"], "r").attrs["voxel_size"]]
    samples = []
    for name, v in volumes.items():
        if "labels" in v:
            mask = training_mask(v, os.path.join(round_dir, "training_mask.zarr", name))
            samples.append({"raw": v["raw"], "labels": v["labels"], **({"mask": mask} if mask else {})})

    done, outputs = [], {name: {"raw": v["raw"]} for name, v in volumes.items()}  # (model, iteration) so far; each volume's latest arrays by name
    n = 1
    for model in chain:
        if model.shipped:
            setup_dir = os.path.join(BS_DIR, "models", model.name)
            found = find_checkpoints(setup_dir)
            checkpoint = found[max(found)] if found else os.path.join(setup_dir, f"model_checkpoint_{SHIPPED_ITERATION}")
        else:
            setup_dir = os.path.join(round_dir, model.name)
            copy_model_scripts(model.name, setup_dir, cli_edit=False)
            train = {"setup_dir": setup_dir, "voxel_size": voxel_size, "max_iterations": 30001,
                     "save_checkpoints_every": 5000, "save_snapshots_every": 1000, "devices": "auto",
                     "samples": samples}
            write(f"{n:02d}_train_{model.name}.toml", {"train": {model.name: train}})
            n += 1
            checkpoint = os.path.join(setup_dir, f"model_checkpoint_{train['max_iterations'] - 1}.ckpt")
        iteration = checkpoint_iteration(checkpoint)
        chain_str = "--from--".join(f"{s}_{it}" for s, it in done)
        predict = {"setup_dir": setup_dir, "checkpoint": checkpoint, "num_gpus": num_gpus, "num_workers": num_gpus}
        for name, v in volumes.items():
            prefix = os.path.join(v.get("outputs", v["raw"].rsplit(".zarr", 1)[0] + ".zarr"), model.name)
            level = f"{iteration}--from--{chain_str}" if chain_str else str(iteration)
            predict[name] = {"input_datasets": [outputs[name][x] for x in model.inputs], "output_datasets_prefix": prefix, "chain_str": chain_str}
            outputs[name] = {o: os.path.join(prefix, level, o) for o in model.outputs}
        write(f"{n:02d}_predict_{model.name}.toml", {"predict": {model.name: predict}})
        n += 1
        done.append((model.name, iteration))

    neighborhood = net_config(setup_dir)["outputs"]["3d_affs"]["neighborhood"]
    params = dict(SEG_DEFAULTS["mws"], aff_neighborhood=neighborhood)
    segment = {"method": "mws", "blockwise": True, "num_workers": 10, "params": params}
    refine, evaluate = {"fill-holes": {"xy": True}}, {}
    for name, v in volumes.items():
        affs = outputs[name]["3d_affs"]
        level = os.path.dirname(affs)
        segment[name] = {"affs_dataset": affs, "fragments_dataset": os.path.join(level, "fragments_mws"),
                         "lut_dir": os.path.join(level, "luts_mws"), "seg_dataset_prefix": os.path.join(level, "segmentations_mws"),
                         **({"mask_dataset": v["target_mask"]} if "target_mask" in v else {})}
        refine[name] = {"in_array": os.path.join(level, "segmentations_mws", mws_segmentation_name(params))}
        if "gt" in v:
            evaluate[name] = {"seg_datasets_prefix": os.path.join(level, "segmentations_mws"),
                              **({"mask_dataset": v["target_mask"]} if "target_mask" in v else {}),
                              "gt": {"labels_dataset": v["gt"]},
                              "pred": {"pred_dataset": affs, "thresholds": EVAL_ERROR_THRESHOLDS["affs"], "params": {"aff_neighborhood": neighborhood}}}
    write(f"{n:02d}_segment.toml", {"segment": segment})
    write(f"{n + 1:02d}_refine.toml", {"refine": refine})
    n += 2
    if evaluate:
        write(f"{n:02d}_evaluate.toml", {"evaluate": evaluate})
    return written


@click.command()
@click.option("--out", "-o", "round_dir", type=click.Path(), help="Round directory to create (asked if not given)")
def prepare(round_dir):
    """Ask about your data and what you want, then write the round directory:
    volumes.toml, numbered step files, and the setup to train."""
    round_dir = os.path.abspath(round_dir or cli_prompt("Round directory", STYLE, default=os.path.join(os.getcwd(), "round_01")))
    os.makedirs(round_dir, exist_ok=True)
    volumes = {}
    while True:
        click.echo()
        name, volume = ask_volume(round_dir)
        volumes[name] = volume
        if not cli_confirm("Another volume?", STYLE, default=False):
            break
    chain = ask_models(volumes)
    num_gpus = ask_gpus()
    written = write_round(round_dir, volumes, chain, num_gpus)
    click.echo()
    for path in written:
        cli_echo(f"wrote {path}", STYLE)
    cli_echo(f"Edit a file to change a parameter. Then: bs run {round_dir}", STYLE, "success")
