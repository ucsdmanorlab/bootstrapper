"""bs prepare: ask the facts of your data in a short dialogue, or take them as flags, then
write the round; on a round, add the volume that flags give and write its run tails again."""
import glob
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field, replace
from typing import Callable

import click
import zarr

from . import discover
from .catalog import check_round, default_chain, describe, download_fix, parse_chain, valid_chains
from .config import RESERVED, load_round, load_volumes, volume_name
from .configs import MODEL_DIR, copy_model_scripts, download_checkpoints, find_checkpoints
from .data.convert import convert
from .data.mask import training_mask, write_training_mask
from .derive import derive
from .params import defaults
from .styles import cli_echo
from .write import file_text, splice, tail_text

STYLE = "prepare"


@dataclass(frozen=True)
class Draft:
    """One volume's answers so far, and what bs prepare read to ask the next question."""
    keys: dict = field(default_factory=dict)  # the volume's keys, as volumes.toml holds them
    name: str = ""
    raw: discover.Array = None  # the raw, or the first array its glob matches
    arrays: tuple = ()  # the arrays of the raw's container
    images: tuple = ()  # a container's image arrays, until an answer picks one
    source: str = ""  # a tif or a folder of 2D images, until it converts into the round
    labels_read: str = ""  # what the 2D check read
    taken: frozenset = frozenset()  # the names of the round's other volumes
    round: str = ""  # the folder an image source converts into


@dataclass(frozen=True)
class Answers:
    """The facts of a round: each volume's keys by name, and the models in run order."""
    volumes: dict
    chain: tuple = ()


@dataclass(frozen=True)
class Question:
    flag: str  # answers it from the command line; None: only the dialogue asks it
    prompt: str
    help: str  # its first line and the example make the flag's help; '?' prints the lines below it
    example: str
    parse: Callable  # (text, state) -> the next state; raises click.ClickException with the fix
    default: Callable = None  # state -> the text Enter gives; None: a fact, which has none
    when: Callable = None  # state -> whether the question applies
    show: Callable = None  # state -> the choices printed above the prompt
    need: Callable = None  # state -> the error for a fact that no flag gives and no terminal asks
    note: Callable = None  # state -> a line printed after the answer


def parse_raw(text, d):
    path = os.path.abspath(os.path.expanduser(text))
    first = min(glob.glob(path), default=None) if glob.has_magic(path) else path
    if first and discover.is_array(first):
        container = discover.container(first)
        return replace(d, keys={"raw": path}, raw=discover.read(first, container), arrays=tuple(discover.arrays(container)))
    if first and ".zarr" in first and os.path.isdir(first):
        arrays = tuple(discover.arrays(discover.container(first)))
        images = tuple(a for a in arrays if a.role == "image" and a.path.startswith(first + os.sep))
        if not images:
            raise click.ClickException(f"{text} holds no image array")
        d = replace(d, keys={"raw": discover.container(path)}, arrays=arrays, images=images)
        return pick_raw("1", d) if len(images) == 1 else d
    if first and os.path.exists(first):
        return replace(d, source=first)
    raise click.ClickException(f"{text}: no zarr array or container, tif or image folder there")


def pick_raw(text, d):
    container = discover.container(d.images[0].path)
    path = os.path.join(container, text)
    raw = _numbered(text, d.images) or (discover.read(path, container) if discover.is_array(path) else None)
    if not raw:
        raise click.ClickException(f"{text}: give a number from the list, or an array in {container}")
    return replace(d, keys={"raw": os.path.join(d.keys["raw"], raw.name)}, raw=raw, images=())


def parse_voxel_size(text, d):
    parts = re.split(r"[\s,]+", text.strip())
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise click.ClickException(f"{text}: give 3 numbers in nm, z y x, such as 50 4 4")
    if not d.source:
        return replace(d, keys=d.keys | {"voxel_size": [int(p) for p in parts]})
    out = os.path.join(d.round, os.path.splitext(os.path.basename(d.source))[0] + ".zarr", "raw")
    click.get_current_context().invoke(convert, in_path=d.source, out_array=out, voxel_size=tuple(map(int, parts)))
    return parse_raw(out, replace(d, source=""))


def parse_labels(text, d):
    if not text:
        return d
    path, labels = _array(text, d, _candidates(d, "labels"))
    flat, read = labels_2d(labels)
    return replace(d, keys=d.keys | {"labels": path, "2d_labels": flat}, labels_read=read)


def labels_2d(labels):
    """2D when no sampled labelled section has a labelled next section. Reads the coarsest scale:
    the answer is the same, and the read smaller."""
    coarse = max((s for s in discover.scales(labels) if s.dtype == labels.dtype and s.voxel_size[0] == labels.voxel_size[0]), key=lambda s: s.voxel_size)
    arr = zarr.open_array(coarse.path, mode="r")
    n = arr.shape[0]
    zs = range(0, n, max(1, n // 16))
    labelled = [z for z in zs if arr[z].any()]
    stacked = sum(z + 1 < n and bool(arr[z + 1].any()) for z in labelled)
    return not stacked, (f"{labels.name}: labels on {len(labelled)} of {len(zs)} sampled sections, {stacked} with "
                         f"labels on the next section, so {'3D' if stacked else '2D'} labels")


LABELLING = {"1": "labels", "sparsely": "labels", "2": "everywhere", "densely": "everywhere", "3": ""}


def parse_labelling(text, d):
    if text in LABELLING:
        return replace(d, keys=d.keys | {"training_mask": LABELLING[text]})
    try:
        return parse_mask(text, d)
    except click.ClickException as e:
        raise click.ClickException(f"answer sparsely, densely, or a mask path; {e.message}")


def parse_mask(text, d):
    return replace(d, keys=d.keys | {"training_mask": _array(text, d)[0]})


def parse_untraced(text, d):
    if text in ("o", "other"):
        return d
    if text in ("b", "background"):
        return replace(d, keys=d.keys | {"mask_dilation": 10})
    raise click.ClickException(f"{text}: answer o (other objects) or b (background)")


def parse_target_mask(text, d):
    return replace(d, keys=d.keys | {"target_mask": _array(text, d)[0]}) if text else d


def parse_gt(text, d):
    if not text:
        return d
    if text == "labels" and "labels" in d.keys:
        return replace(d, keys=d.keys | {"gt": d.keys["labels"]})
    return replace(d, keys=d.keys | {"gt": _array(text, d)[0]})


def parse_name(text, d):
    if text in RESERVED:
        raise click.ClickException(f"a volume cannot be named {', '.join(RESERVED[:-1])} or {RESERVED[-1]}. Example: {text}-1")
    if text in d.taken:
        raise click.ClickException(f"{text} names another volume of this round. Example: {text}-2")
    return replace(d, name=text)


def parse_models(text, answers):
    chain = parse_chain(text)
    check_round(chain)
    return replace(answers, chain=tuple(chain))


def _array(text, d, choices=()):
    """The array an answer names: its number in choices, a path in the raw's container, or a path
    from the working folder. Returns the path to write, a glob like the raw's, and the array."""
    container = discover.container(d.raw.path)
    here = os.path.normpath(os.path.join(container, text))
    path = here if discover.is_array(here) else os.path.abspath(os.path.expanduser(text))
    array = _numbered(text, choices) or (discover.read(path, container) if discover.is_array(path) else None)
    if not array:
        raise click.ClickException(f"{text}: no zarr array there or in {container}")
    if array.voxel_size != d.raw.voxel_size:
        same = [s for s in discover.scales(array) if s.voxel_size == d.raw.voxel_size]
        use = f" Use {os.path.join(os.path.dirname(text), os.path.basename(same[0].path))}." if same else ""
        raise click.ClickException(f"{text} has voxel {_nm(array.voxel_size)} nm; the raw has {_nm(d.raw.voxel_size)} nm.{use}")
    return array.path.replace(container, discover.container(d.keys["raw"]), 1), array


def _numbered(text, choices):
    return choices[int(text) - 1] if text.isdigit() and 0 < int(text) <= len(choices) else None


def _candidates(d, role):
    return discover.candidates(d.raw, d.arrays)[role]


def _nm(values):
    return " ".join(map(str, values))


def _rows(arrays):
    """    1  raw/s1  124 4926 9750  uint8  50 4 4 nm"""
    rows = [(a.name, _nm(a.shape), a.dtype, f"{_nm(a.voxel_size)} nm" + (f"  offset {_nm(a.offset)}" if any(a.offset) else "")) for a in arrays]
    widths = [max(len(row[i]) for row in rows) for i in range(3)] + [0]
    return "\n".join(f"    {n}  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for n, row in enumerate(rows, 1))


def _masks(d):
    masks = _candidates(d, "mask")
    if masks:
        return (f"  {len(masks)} mask{'s' * (len(masks) != 1)} at {_nm(d.raw.voxel_size)} nm, such as "
                f"{', '.join(a.name for a in masks[:3])}. Tab completes.")


def _need_pick(d):
    images = ", ".join(f"{a.name} ({_nm(a.voxel_size)} nm)" for a in d.images)
    return (f"--raw {d.keys['raw']} holds {len(d.images)} image arrays; give one: {images}. "
            f"Example: --raw {d.keys['raw']}/{d.images[0].name}")


def _no_voxel_size(d):
    return bool(d.source) or "voxel_size" not in zarr.open_array(d.raw.path, mode="r").attrs


VOLUME = [
    Question("--raw", "Raw image (zarr array or container, tif, image folder; globs work)",
             "The image to segment: a zarr array or container, a tif, or a folder of 2D images; a glob makes one volume per match.",
             "data.zarr/raw/s0", parse_raw,
             need=lambda d: "--raw is needed: a zarr array or container, a tif, or a folder of 2D images. Example: --raw data.zarr/raw/s0"),
    Question(None, "Raw array", "The image array to segment: its number in the list, or its path in the container.", "",
             pick_raw, when=lambda d: bool(d.images), show=lambda d: _rows(d.images), need=_need_pick),
    Question("--voxel-size", "Voxel size in nm, z y x", "The raw's voxel size in nm, z y x, when its metadata has none.",
             "50,4,4", parse_voxel_size, when=_no_voxel_size,
             need=lambda d: f"--voxel-size is needed: {d.source or d.raw.path} has no voxel size. Example: --voxel-size 50,4,4"),
    Question("--labels", "Labels (Enter: none, only predict)",
             "Traced objects to train on, relative to the raw's container or absolute. Without labels the volume is only predicted.",
             "labels/s0", parse_labels, default=lambda d: "", show=lambda d: _rows(c) if (c := _candidates(d, "labels")) else None,
             note=lambda d: d.labels_read),
    Question("--labelled", "Labelling",
             "How the labels cover the raw: sparsely, densely, or a mask path (densely labelled within it).\n"
             "sparsely labelled: some objects are traced; training learns only from them\n"
             "densely labelled: every object is traced; training also learns the space between them\n"
             "densely labelled within a mask: every object inside the mask is traced",
             "sparsely", parse_labelling, when=lambda d: "labels" in d.keys,
             show=lambda d: "    1  sparsely labelled\n    2  densely labelled\n    3  densely labelled within a mask",
             need=lambda d: "--labelled is needed with --labels: sparsely, densely, or a mask path (densely labelled within it). "
                            "Example: --labelled sparsely"),
    Question(None, "Mask", "The mask inside which every object is traced. It becomes the training mask.", "", parse_mask,
             when=lambda d: d.keys.get("training_mask") == "", show=_masks,
             need=lambda d: "--labelled takes the mask's path. Example: --labelled mask/s0"),
    Question("--untraced", "Untraced space next to a traced object: [o]ther objects / [b]ackground",
             "With sparse labels, what the untraced space next to a traced object is: other (objects) or background.\n"
             "other objects: as in EM neuropil, where untraced cells touch the traced ones\n"
             "background: as in cells on an empty slide; training learns 10 voxels of it around each object",
             "background", parse_untraced, default=lambda d: "o", when=lambda d: d.keys.get("training_mask") == "labels"),
    Question("--target-mask", "Target mask (Enter: none, segment everywhere)",
             "Segment, refine and evaluate only inside this mask; predict covers its box.", "mask/s0", parse_target_mask,
             default=lambda d: "", show=_masks),
    Question("--gt", "Ground truth ('labels' reuses the labels; Enter: none)",
             "Ground truth for evaluate; 'labels' reuses the labels. Without it, the round has no evaluate step.", "labels",
             parse_gt, default=lambda d: ""),
    Question("--name", "Volume name", "The volume's name, which names its tables, such as [segment.apical]. Default: the "
             "container's file name, such as data-zarr.", "apical", parse_name,
             default=lambda d: volume_name(d.raw.path, d.taken), when=lambda d: not glob.has_magic(d.keys["raw"])),
]
VOLUME_FLAGS = {q.flag for q in VOLUME if q.flag}
CHAINS = valid_chains()
MODELS = Question(
    "--models", f"Models (? lists all {len(CHAINS)})",
    "The models in run order, by name or short form; the default fits the labels.\n"
    + "".join(f"{' '.join(m.short for m in chain):9s} {describe(chain)}\n" for chain in CHAINS)
    + "Type names or short forms, such as 2M 3Af2L.",
    "2M,3Af2L", parse_models,
    default=lambda a: " ".join(m.name for m in default_chain(any(v.get("2d_labels") for v in a.volumes.values()))),
    note=lambda a: describe(a.chain))


def ask(questions, state, flags, terminal):
    """The state after each question that applies, answered by its flag, at the terminal, or by its
    default. At the terminal '?' explains a question and '<' goes back one answer."""
    back, shown, i = [], None, 0
    while i < len(questions):
        q = questions[i]
        if q.when and not q.when(state):
            i += 1
            continue
        default = q.default(state) if q.default else None
        text, asked = flags.get(q.flag), False
        if text is None and terminal:
            if q.show and shown != i and (choices := q.show(state)):
                click.echo(choices)
            shown, asked = i, True
            _tab(discover.container(state.raw.path) if getattr(state, "raw", None) else "")
            text = click.prompt(q.prompt, default=default, show_default=bool(default), prompt_suffix=" > ").strip()
            if text in ("?", "<"):
                if text == "?":
                    click.echo("\n".join(f"  {line}" for line in q.help.splitlines()[1:] or [q.help]))
                elif back:
                    i, state = back.pop()
                continue
        elif text is None:
            if default is None:
                raise click.ClickException(q.need(state))
            text = default
        try:
            new = q.parse(text, state)
        except click.ClickException as e:
            if not asked:
                raise click.ClickException(f"{q.flag}: {e.message}" if q.flag in flags else e.message)
            cli_echo(f"  {e.message}", STYLE, "error")
            continue
        if asked:
            back.append((i, state))
        if q.note and (note := q.note(new)):
            cli_echo(f"  {note}", STYLE)
        state, i = new, i + 1
    return state


def ask_volumes(round_dir, flags, terminal):
    """The volumes' keys by name: one volume from flags, or as many as the dialogue gives."""
    volumes = {}
    while True:
        d = ask(VOLUME, Draft(taken=frozenset(volumes), round=round_dir), flags, terminal)
        name = d.name or volume_name(d.keys["raw"], volumes)
        volumes[name] = d.keys
        cli_echo(f"  {summary(name, d)}", STYLE)
        if (not terminal or flags.keys() & VOLUME_FLAGS
                or not click.prompt("Another volume?", default="n", prompt_suffix=" > ").startswith("y")):
            break
    if not any("labels" in v for v in volumes.values()):
        labels = ", ".join(a.name for a in _candidates(d, "labels"))
        where = f"; label arrays at {_nm(d.raw.voxel_size)} nm in {os.path.basename(discover.container(d.raw.path))}: {labels}"
        raise click.ClickException(f"no volume has labels, so no model can train. Give --labels{where if labels else ''}")
    return volumes


def summary(name, d, taken=()):
    """dw09-zarr: raw/s1, sparsely labelled gt_boundaries/s1 (3D; other objects around them), ground truth gt_boundaries/s1"""
    keys, container = d.keys, discover.container(d.keys["raw"])

    def rel(path):
        return os.path.relpath(path, container) if path.startswith(container + os.sep) else path

    parts = [rel(keys["raw"])]
    if "labels" in keys:
        mode = keys["training_mask"]
        around = ("; background around objects" if "mask_dilation" in keys else "; other objects around them") if mode == "labels" else ""
        within = "" if mode in ("labels", "everywhere") else f" within {rel(mode)}"
        parts.append(f"{'sparsely' if mode == 'labels' else 'densely'} labelled {rel(keys['labels'])} "
                     f"({'2D' if keys['2d_labels'] else '3D'}{around}){within}")
    parts += [f"{label} {rel(keys[k])}" for k, label in (("target_mask", "target mask"), ("gt", "ground truth")) if k in keys]
    if glob.has_magic(keys["raw"]):
        names = []
        for match in sorted(glob.glob(keys["raw"])):
            names.append(volume_name(match, {*taken, *names}))
        name = f"{len(names)} volumes ({', '.join(names)})"
    return f"{name}: {', '.join(parts)}"


def _terminal():
    return sys.stdin.isatty()


def _tab(base):
    """Tab completes a path, from base or absolute."""
    import readline

    def complete(text, i):
        full = os.path.join(base, os.path.expanduser(text))
        hits = [text + h[len(full):] + "/" * (os.path.isdir(h) and not discover.is_array(h))
                for h in sorted(glob.glob(glob.escape(full) + "*"))]
        return hits[i] if i < len(hits) else None

    readline.set_completer_delims(" \t\n")
    readline.set_completer(complete)
    readline.parse_and_bind("bind ^I rl_complete" if "libedit" in (readline.__doc__ or "") else "tab: complete")


EVALUATE = ("Compare the segmentations with the ground truth, and the affinities with the segmentations.",
            {"evaluate": defaults("evaluate")})


def write_round(path, answers):
    """Write volumes.toml, unless it is there, and one numbered file of your keys per step; a path
    ending in .toml gets them all in one file. Then write their run tails. Returns the files written."""
    volumes, chain = answers.volumes, answers.chain
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
        files.append(("evaluate", *EVALUATE))
    texts = {f"{n:02d}_{name}.toml" if n else name: file_text(title, doc) for n, (name, title, doc) in enumerate(files)}
    folder = path
    if path.endswith(".toml"):
        folder, texts = os.path.dirname(path), {os.path.basename(path): "\n".join(texts.values())}
    elif os.path.exists(os.path.join(path, "volumes.toml")):
        del texts["volumes.toml"]
    os.makedirs(folder or ".", exist_ok=True)
    for name, text in texts.items():
        with open(os.path.join(folder, name), "w") as f:
            f.write(text)
    refresh(path)
    return [os.path.join(folder, name) for name in texts]


def add_volume(path, flags, terminal):
    """Add the volume that flags give to the round folder at path, in its volumes.toml."""
    if not os.path.isdir(path):
        raise click.ClickException(f"{path} holds the round in one file. Add the volume as a [volumes.<name>] "
                                   f"table in it, then run: bs prepare {path}")
    beside = os.path.join(path, "volumes.toml")
    if "--raw" not in flags:
        raise click.ClickException(f"--raw is needed to add a volume to {path}. To change a volume, edit {beside}, "
                                   f"then run: bs prepare {path}")
    with open(beside, "rb") as f:
        taken = set(tomllib.load(f).get("volumes", {})) | set(load_volumes(beside))
    d = ask(VOLUME, Draft(round=os.path.abspath(path)), flags, terminal)
    name = d.name or volume_name(d.keys["raw"], ())
    if name in taken:
        raise click.ClickException(f"{path} has a volume named {name} already. Give this one another name, such as --name {name}-2")
    cli_echo(f"  {summary(name, d, taken)}", STYLE)
    with open(beside, "a") as f:
        f.write(file_text("", {"volumes": {name: d.keys}}).split("\n", 1)[1])
    cli_echo(f"added {name} to {beside}", STYLE)


def add_steps(path):
    """Write the step files that the volumes need and the round folder at path lacks: an
    evaluate step once a volume has ground truth. Returns the files written."""
    if not os.path.isdir(path):
        return []
    files = load_round(path, tail=False)
    if (not any("gt" in v.keys for v in load_volumes(files[0][0]).values())
            or any(s.kind == "evaluate" for _, steps in files for s in steps)):
        return []
    n = int(re.match(r"\d+", os.path.basename(files[-1][0]))[0]) + 1
    out = os.path.join(path, f"{n:02d}_evaluate.toml")
    with open(out, "w") as f:
        f.write(file_text(*EVALUATE))
    return [out]


def stale(path):
    """(file, its text, its new run tail, {table: {key: (old, new)}}) for each file of the round
    at path whose run tail differs from the one derived from the volumes and the keys above it."""
    files, tails = _derived(path)
    tails = iter(tails)
    out = []
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
            out.append((f, text, run, keys))
    return out


def _derived(path):
    """The files of the round at path with your keys alone, and the run tail derived for each step."""
    root = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path))
    files = load_round(path, tail=False)
    return files, derive(load_volumes(files[0][0]), [s for _, steps in files for s in steps], root)


def fetch_checkpoints(path):
    """Download each shipped checkpoint that a predict step of the round at path reads and
    MODEL_DIR does not have."""
    files, tails = _derived(path)
    for s, tail in zip([s for _, steps in files for s in steps], tails):
        setup_dir = os.path.join(MODEL_DIR, s.name)
        if s.kind == "predict" and tail.get("checkpoint", "").startswith(setup_dir + os.sep) and not find_checkpoints(setup_dir):
            try:
                download_checkpoints(s.name, setup_dir)
            except click.ClickException as e:
                raise click.ClickException(f"{e.message}; {download_fix(s.name, path)}") from e


def refresh(path):
    """Download the shipped checkpoints the round lacks, then write the run tails that stale
    finds. Then copy the model scripts that are missing, and write the training masks. Returns
    (file, {table: {key: (old, new)}}) for each file whose tail changed."""
    root = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path))
    fetch_checkpoints(path)
    changed = stale(path)
    for f, text, run, _ in changed:
        with open(f, "w") as fh:
            fh.write(splice(text, tail_text(path, run) if run else ""))
    files = load_round(path)
    for _, steps in files:
        for s in steps:
            if s.kind == "train" and not os.path.exists(os.path.join(s.keys["setup_dir"], "train.py")):
                copy_model_scripts(s.name, s.keys["setup_dir"], cli_edit=False)
    for name, v in load_volumes(files[0][0]).items():
        out = os.path.join(root, "training_mask.zarr", name)
        if "labels" in v.keys and training_mask(v.keys, out) == out:
            write_training_mask(v.keys, out)
    return [(f, keys) for f, _, _, keys in changed]


def _changed(old, new, table="run"):
    """The keys whose values differ between two run tails, with both values, by table."""
    keys = {}
    for k in [*new, *(k for k in old if k not in new)]:
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) or isinstance(b, dict):
            keys |= _changed(a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {}, f"{table}.{k}")
        elif a != b:
            keys.setdefault(table, {})[k] = (a, b)
    return keys


def _flags(command):
    """One option per question that has a flag, with its help and an example."""
    for q in reversed([*VOLUME, MODELS]):
        if q.flag:
            command = click.option(q.flag, multiple=True, help=f"{q.help.splitlines()[0]} Example: {q.flag} {q.example}")(command)
    return command


@click.command()
@click.argument("path", type=click.Path())
@_flags
def prepare(path, **given):
    """Write the round PATH from the facts of your data: a short dialogue asks them, or flags give
    them. A PATH ending in .toml holds the round in one file. On a round, add the volume that
    flags give, then write each file's run tail again from your edits. Example: bs prepare
    round_01 --raw data.zarr/raw/s0 --labels labels/s0 --labelled sparsely"""
    given = {f"--{k.replace('_', '-')}": v for k, v in given.items()}
    for flag, values in given.items():
        if len(values) > 1:
            if flag == "--models":
                raise click.ClickException(f"--models: give it once, such as --models {values[-1]}")
            raws = given["--raw"]
            again = f" {flag} {values[1]}" * (flag != "--raw")
            raise click.ClickException(f"{flag}: give one volume per command; add the next with bs prepare {path} "
                                       f"--raw {raws[1] if len(raws) > 1 else '...'}{again}")
    flags = {flag: values[0] for flag, values in given.items() if values}
    if os.path.isfile(path) or glob.glob(os.path.join(path, "[0-9]*.toml")):
        if flags.keys() - VOLUME_FLAGS:
            raise click.ClickException(f"--models: {path} is a round already. Edit its files, then run: bs prepare {path}")
        if flags:
            add_volume(path, flags, _terminal())
        added = add_steps(path)
        for f in added:
            cli_echo(f"wrote {f}", STYLE)
        for f, keys in refresh(path):
            tables = [f"[{table}] {', '.join(k)}" for table, k in keys.items()]
            more = f" and {len(tables) - 3} more tables" if len(tables) > 3 else ""
            if f not in added:
                cli_echo(f"refreshed {os.path.basename(f)}: {'; '.join(tables[:3])}{more}", STYLE)
        cli_echo(f"{path} is current.", STYLE, "success")
        cli_echo(f"Next: bs run {path}", STYLE)
        return
    if ".zarr" in path:
        raise click.ClickException(f"{path} is zarr data, not a round. Give a new round folder and the raw: bs prepare round_01 --raw {path}")
    terminal, beside = _terminal(), os.path.join(path, "volumes.toml")
    if os.path.exists(beside):
        if flags.keys() & VOLUME_FLAGS:
            raise click.ClickException(f"{beside} holds the volumes; edit it, or give a new round folder")
        volumes = {name: v.keys for name, v in load_volumes(beside).items()}
    else:
        if not terminal and not flags:
            raise click.ClickException("stdin is not a terminal, so bs prepare asks nothing. Give the facts as flags, "
                                       "such as --raw data.zarr/raw/s0")
        if terminal and not flags:
            cli_echo("Set up a round: your arrays, then the models. '?' explains a question, '<' goes back, Tab completes.", STYLE)
        volumes = ask_volumes(os.path.abspath(os.path.dirname(path) if path.endswith(".toml") else path), flags, terminal)
    for f in write_round(path, ask([MODELS], Answers(volumes), flags, terminal)):
        cli_echo(f"wrote {f}", STYLE)
    cli_echo(f"Next: bs run {path}\nbs run {path} --dry-run checks every path and runs nothing.\n"
             "To change a param, edit its file; every key says what it does.", STYLE, "success")
