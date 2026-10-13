"""bs prepare: ask about the data, then write the round directory; on a round, write its
run tails again."""
import glob
import os
import tomllib

import click
import zarr

from .catalog import check_round, default_chain, describe, parse_chain, valid_chains
from .config import load_round, load_volumes
from .configs import copy_model_scripts
from .data.convert import convert
from .data.mask import training_mask, write_training_mask
from .derive import derive
from .params import defaults
from .styles import cli_echo, cli_prompt, cli_confirm
from .write import file_text, splice, tail_text

STYLE = "prepare"


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


def write_round(round_dir, volumes, chain):
    """Write volumes.toml and one numbered file of your keys per step, then their run tails."""
    if not any("labels" in v for v in volumes.values()):
        raise click.ClickException("no volume has labels; nothing to train")
    files = [("volumes.toml", "Volumes of this round. Every step reads its arrays from here.", {"volumes": volumes})]
    for model in chain:
        io = f"{' and '.join(model.inputs)} in, {' and '.join(model.outputs)} out"
        if not model.shipped:
            files.append((f"train_{model.name}", f"Train {model.name} ({model.short}) on every volume with labels: {io}.",
                          {"train": {model.name: defaults("train.*")}}))
        files.append((f"predict_{model.name}", f"Predict {model.name} ({model.short}) on every volume: {io}.",
                      {"predict": {model.name: defaults("predict.*")}}))
    files.append(("segment", f"Segment the 3D affinities of {chain[-1].name} into objects.",
                  {"segment": defaults("segment") | {"params": defaults("segment.params")}}))
    masked = {"mask": {}} if all("target_mask" in v for v in volumes.values()) else {}
    files.append(("refine", "Refine the segmentation. The ops run in file order; each output feeds the next.",
                  {"refine": {"fill-holes": defaults("refine.fill-holes")} | masked}))
    if any("gt" in v for v in volumes.values()):
        files.append(("evaluate", "Compare the segmentations with the ground truth, and the affinities with the segmentations.",
                      {"evaluate": defaults("evaluate")}))
    written = []
    for n, (name, title, doc) in enumerate(files):
        path = os.path.join(round_dir, f"{n:02d}_{name}.toml" if n else name)
        with open(path, "w") as f:
            f.write(file_text(title, doc))
        written.append(path)
    refresh(round_dir)
    return written


def refresh(path):
    """Write the run tail of each file of the round at path from the volumes and the keys
    above it. Then copy the model scripts that are missing, and write the training masks.
    Returns (file, {table: keys}) for each file whose tail changed."""
    root = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path))
    files = load_round(path, tail=False)
    volumes = load_volumes(files[0][0])
    tails = iter(derive(volumes, [s for _, steps in files for s in steps], root))
    changed = []
    for f, steps in files:
        run = {}
        for s, table in zip(steps, tails):
            if table and s.name == s.kind:
                run[s.kind] = table
            elif table:
                run.setdefault(s.kind, {})[s.name] = table
        with open(f) as fh:
            text = fh.read()
        if keys := _changed(tomllib.loads(text).get("run", {}), run):
            changed.append((f, keys))
            with open(f, "w") as fh:
                fh.write(splice(text, tail_text(os.path.abspath(path), run) if run else ""))
    for _, steps in load_round(path):
        for s in steps:
            if s.kind == "train" and not os.path.exists(os.path.join(s.keys["setup_dir"], "train.py")):
                copy_model_scripts(s.name, s.keys["setup_dir"], cli_edit=False)
    for name, v in volumes.items():
        out = os.path.join(root, "training_mask.zarr", name)
        if "labels" in v.keys and training_mask(v.keys, out) == out:
            write_training_mask(v.keys, out)
    return changed


def _changed(old, new, table="run"):
    """The keys whose values differ between two run tails, by table."""
    keys = {}
    for k in [*new, *(k for k in old if k not in new)]:
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) or isinstance(b, dict):
            keys |= _changed(a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {}, f"{table}.{k}")
        elif a != b:
            keys.setdefault(table, []).append(k)
    return keys


@click.command()
@click.argument("path", type=click.Path())
def prepare(path):
    """Ask about your data and what you want, then write the round directory PATH:
    volumes.toml, numbered step files, and the setup to train. On a round, write the run
    tail of each file again from your edits, such as: bs prepare round_01"""
    if os.path.isfile(path) or glob.glob(os.path.join(path, "[0-9]*.toml")):
        for f, keys in refresh(path):
            tables = [f"[{table}] {', '.join(k)}" for table, k in keys.items()]
            more = f" and {len(tables) - 3} more tables" if len(tables) > 3 else ""
            cli_echo(f"refreshed {os.path.basename(f)}: {'; '.join(tables[:3])}{more}", STYLE)
        cli_echo(f"{path} is current.", STYLE, "success")
        return
    round_dir = os.path.abspath(path)
    os.makedirs(round_dir, exist_ok=True)
    volumes = {}
    while True:
        click.echo()
        name, volume = ask_volume(round_dir)
        volumes[name] = volume
        if not cli_confirm("Another volume?", STYLE, default=False):
            break
    chain = ask_models(volumes)
    written = write_round(round_dir, volumes, chain)
    click.echo()
    for f in written:
        cli_echo(f"wrote {f}", STYLE)
    cli_echo(f"Edit a file to change a parameter, then: bs prepare {round_dir}. To run the round: bs run {round_dir}", STYLE, "success")
