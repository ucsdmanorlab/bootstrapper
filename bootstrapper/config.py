"""Config files are kind tables. [volumes.<name>] describes data; [train.<name>],
[predict.<name>], [segment], [segment.<name>], [refine] and [evaluate] are steps. A file
holds one step or a whole round; a step is named only when a kind has several. Below your
keys, a run tail of [run.<step>] tables holds the keys bs prepare derives."""
import difflib
import glob
import os
import tomllib
from dataclasses import dataclass, field

import click

KINDS = ("train", "predict", "segment", "refine", "evaluate")
STEP_TABLES = {"params", "db", "pred", "gt", "samples"}
OPS = ("size-filter", "outlier-filter", "z-filter", "remap", "fill-holes", "dilate",
       "erode", "opening", "closing", "mask")
OLD_PARAMS = {"ws_params", "mws_params", "cc_params"}
POSTGRES = {"db_name", "db_host", "db_user", "db_password"}
RESERVED = (*KINDS, "volumes", "run", "round")
VOLUME_KEYS = {"raw", "labels", "training_mask", "2d_labels", "mask_dilation", "target_mask", "gt", "outputs", "voxel_size"}
OLD_VOLUME_KEYS = {
    "sparse_labels": 'write training_mask = "labels" (sparsely labelled) or "everywhere" (densely labelled)',
    "dense_objects": "delete it, or write mask_dilation = 10 if untraced space next to an object is background",
    "mask": "rename it target_mask",
}
KEYS = {
    "predict": {"setup_dir", "checkpoint", "input_datasets", "output_datasets_prefix",
                "chain_str", "num_workers", "num_gpus", "roi_offset", "roi_shape"},
    "segment": {"method", "affs_dataset", "fragments_dataset", "lut_dir", "seg_dataset_prefix",
                "mask_dataset", "seeds_dataset", "blockwise", "block_shape", "context",
                "num_workers", "roi_offset", "roi_shape"},
    "refine": {"in_array", "num_workers"},
    "evaluate": {"seg_datasets_prefix", "seg_datasets", "mask_dataset", "out_result", "out_dir",
                 "thresholds", "roi_offset", "roi_shape"},
}
READS = {  # the keys that name what a step reads
    "train": ("raw", "labels", "mask"),
    "predict": ("input_datasets", "checkpoint"),
    "segment": ("affs_dataset", "mask_dataset", "seeds_dataset"),
    "refine": ("in_array", "mask_array"),
    "evaluate": ("seg_datasets_prefix", "seg_datasets", "labels_dataset", "pred_dataset", "mask_dataset"),
}
WRITES = {  # the keys that name where a step writes; refine writes beside its in_array
    "train": ("setup_dir",),
    "predict": ("output_datasets_prefix",),
    "segment": ("fragments_dataset", "lut_dir", "seg_dataset_prefix"),
    "refine": ("in_array",),
    "evaluate": ("out_result", "out_dir"),
}


@dataclass
class Volume:
    name: str
    keys: dict  # raw, labels, training_mask, target_mask, gt, facts, and arrays under your own names

    def array(self, ref):
        """A path has a slash; a name has none and names one of this volume's arrays."""
        return ref if "/" in ref else self.keys[ref]


@dataclass
class Step:
    kind: str
    name: str
    keys: dict
    volumes: dict = field(default_factory=dict)  # per-volume keys, from [<step>.<volume>]

    @property
    def label(self):
        return self.kind if self.name == self.kind else f"{self.kind}.{self.name}"


def load(path, tail=True):
    """The steps of a file, each over its run tail; with tail=False, your keys alone."""
    doc = _toml(path)
    volumes = load_volumes(path)
    doc.pop("volumes", None)
    doc.pop("round", None)
    run = doc.pop("run", {})
    run = run if tail else {}

    old = next((k for k in doc if k in OLD_PARAMS), None)
    rename = f"rename [{old}] to [segment.params]"
    if any(not isinstance(v, dict) for v in doc.values()):
        raise click.ClickException(
            f"{path}: keys at the top level; wrap them in [segment] (or [evaluate], [train.<name>], "
            "[predict.<name>])" + (f", and {rename}" if old else "")
        )
    if old:
        raise click.ClickException(f"{path}: {rename}")

    steps = []
    for kind, table in doc.items():
        if kind not in KINDS:
            raise click.ClickException(f"{path}: unknown table [{kind}]; tables are volumes, {', '.join(KINDS)}")
        if _is_step(kind, table, volumes):
            steps.append(_step(kind, kind, table, volumes, path, run.get(kind)))
        else:
            for name, sub in table.items():
                if name in volumes:
                    raise click.ClickException(f"{path}: [{kind}.{name}] is a step named like the volume {name!r}")
                steps.append(_step(kind, name, sub, volumes, path, run.get(kind, {}).get(name)))
    return steps


def load_round(path, tail=True):
    """(file, steps) in run order: the numbered files of a folder, or one file."""
    if not os.path.isdir(path):
        return [(path, load(path, tail))]
    files = sorted(glob.glob(os.path.join(path, "[0-9]*.toml")))
    if not files:
        raise click.ClickException(f"{path}: no numbered .toml files")
    return [(f, load(f, tail)) for f in files]


def prepared_round(path):
    """The round that path runs in, if bs prepare wrote it (a file holds a run tail); None for
    files of your own. A step file whose volumes sit in volumes.toml beside it runs in that
    folder's round."""
    beside = os.path.join(os.path.dirname(os.path.abspath(path)), "volumes.toml")
    if os.path.isfile(path) and "volumes" not in _toml(path) and os.path.exists(beside):
        path = os.path.dirname(beside)
    return path if any("run" in _toml(f) for f, _ in load_round(path, tail=False)) else None


def load_volumes(path):
    """The volumes of a file, else of the volumes.toml beside it. Globs give one volume per
    match, named by its container; the globs of a volume pair up in sorted order."""
    tables = _toml(path).get("volumes", {})
    beside = os.path.join(os.path.dirname(os.path.abspath(path)), "volumes.toml")
    if not tables and os.path.exists(beside):
        path, tables = beside, _toml(beside).get("volumes", {})
    taken, volumes = set(tables), {}
    for table, keys in tables.items():
        where = f"{path}: [volumes.{table}]"
        if not isinstance(keys, dict):
            raise click.ClickException(f"{path}: [volumes] {table} is a key, not a volume; write [volumes.<name>] above it")
        if table in RESERVED:
            raise click.ClickException(f"{where}: a volume cannot be named {', '.join(RESERVED[:-1])} or {RESERVED[-1]}. Example: [volumes.{table}-1]")
        old = next((k for k in keys if k in OLD_VOLUME_KEYS), None)
        if old:
            raise click.ClickException(f"{where} has no key {old!r}; {OLD_VOLUME_KEYS[old]}")
        refuse_unknown(where, [k for k, v in keys.items() if k not in VOLUME_KEYS and not _own_array(k, v)],
                       VOLUME_KEYS, ", or a name of your own for an array path")
        volumes |= {name: Volume(name, k) for name, k in _expand(table, keys, where, taken).items()}
    return volumes


def _toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def _own_array(key, value):
    """A path under a key of your own, unless the key looks like a misspelt one: pgt is a
    name, lables a typo of labels."""
    return isinstance(value, str) and all(k in key for k in difflib.get_close_matches(key, VOLUME_KEYS, 1, 0.8))


def _expand(table, keys, where, taken):
    globs = {k: sorted(glob.glob(v)) for k, v in keys.items() if isinstance(v, str) and glob.has_magic(v)}
    if not globs:
        return {table: keys}
    first, matches = next(iter(globs.items()))
    if not matches:
        raise click.ClickException(f"{where} {first} = {keys[first]!r} matches nothing")
    for k, found in globs.items():
        if len(found) != len(matches):
            raise click.ClickException(f"{where} {first} matches {len(matches)} arrays and {k} matches {len(found)}; they must match one to one")
    volumes = {}
    for i, path in enumerate(matches):
        name = _volume_name(path, taken)
        taken.add(name)
        volumes[name] = keys | {k: found[i] for k, found in globs.items()}
    return volumes


def _volume_name(path, taken):
    """The container's file name with the dot as a dash, since a table name takes no dot:
    training.zarr gives training-zarr, and a second volume in it training-zarr-2."""
    container = path.rsplit(".zarr", 1)[0] + ".zarr" if ".zarr" in path else path
    name = base = os.path.basename(container).replace(".", "-")
    n = 2
    while name in taken:
        name, n = f"{base}-{n}", n + 1
    return name


def steps_of(path, kind, name=None):
    """The steps of one kind in a file, or the one named."""
    steps = [s for s in load(path) if s.kind == kind]
    if not steps:
        raise click.ClickException(f"{path}: no [{kind}] step")
    if name:
        names = [s.name for s in steps]
        steps = [s for s in steps if s.name == name]
        if not steps:
            raise click.ClickException(f"{path}: no [{kind}.{name}]; {kind} steps are {', '.join(names)}")
    return steps


def runs(step):
    """One flat config per volume table, else one for the step itself. A volume's
    tables, such as params and db, merge key by key over the step's."""
    if not step.volumes:
        return [(None, dict(step.keys))]
    return [(vol, _merge(step.keys, keys)) for vol, keys in step.volumes.items()]


def paths(config, key):
    """The paths under key in config, its tables and its arrays of tables."""
    for k, v in config.items():
        if isinstance(v, dict):
            yield from paths(v, key)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            for row in v:
                yield from paths(row, key)
        elif k == key:
            yield from v if isinstance(v, list) else [v]


def ops_beside(keys):
    """A refine table's ops beside its keys, as a file holds them."""
    return {k: v for k, v in keys.items() if k != "ops"} | keys.get("ops", {})


def _merge(a, b):
    """b over a; a table in both merges key by key."""
    return a | b | {k: _merge(a[k], b[k]) for k in a.keys() & b.keys() if isinstance(a[k], dict) and isinstance(b[k], dict)}


def with_tail(step, tail):
    """The step over its run tail. Your per-volume table sets a key first, then the step, then the tail."""
    vols = {v: _merge(unset(tail.volumes.get(v, {}), step.keys), step.volumes.get(v, {})) for v in step.volumes | tail.volumes}
    return Step(step.kind, step.name, _merge(tail.keys, step.keys), vols)


def unset(table, *above):
    """The keys of table that no table above sets; a table in both keeps the keys only it has."""
    out = {}
    for k, v in table.items():
        set_above = [a[k] for a in above if k in a]
        if isinstance(v, dict) and all(isinstance(a, dict) for a in set_above):
            v = unset(v, *set_above)
        elif set_above:
            continue
        if v != {}:
            out[k] = v
    return out


def _own(kind, name):
    return name in STEP_TABLES or name in OLD_PARAMS or (kind == "refine" and name in OPS)


def _is_step(kind, table, volumes):
    """A table with a scalar key is one step; so is one whose sub-tables are all its own
    or volumes. Otherwise its sub-tables are named steps."""
    if any(not isinstance(v, dict) for v in table.values()):
        return True
    return all(_own(kind, k) or k in volumes for k in table)


def _step(kind, name, table, volumes, path, tail=None):
    label = kind if name == kind else f"{kind}.{name}"
    keys = dict(table)
    old = OLD_PARAMS & set(keys)
    if old:
        raise click.ClickException(f"{path}: [{label}.{old.pop()}] is now [{label}.params]")
    vols = {k: keys.pop(k) for k in list(keys) if k in volumes}
    if kind == "refine":
        keys["ops"] = {k: keys.pop(k) for k in list(keys) if k in OPS}
        for t in vols.values():
            if ops := {k: t.pop(k) for k in list(t) if k in OPS}:
                t["ops"] = ops
    if kind != "train":
        _check(path, label, kind, keys, volumes)
        for vol, t in vols.items():
            _check(path, f"{label}.{vol}", kind, t, volumes)
    step = Step(kind, name, keys, vols)
    return with_tail(step, _step(kind, name, tail, volumes, path)) if tail else step


def _check(path, label, kind, keys, volumes):
    ops = f"; ops are {', '.join(OPS)}" if kind == "refine" else ""
    tables = [k for k, v in keys.items() if isinstance(v, dict) and k not in STEP_TABLES and k != "ops"]
    if tables:
        vols_known = f"volumes are {', '.join(volumes)}" if volumes else "no volumes are defined"
        raise click.ClickException(f"{path}: [{label}.{tables[0]}] is neither a table of [{label}] nor a volume; {vols_known}{ops}")
    refuse_unknown(f"{path}: [{label}]", [k for k, v in keys.items() if not isinstance(v, dict)], KEYS[kind], ops)
    db = keys.get("db", {})
    if "db_port" in db:
        raise click.ClickException(f"{path}: [{label}.db] has no key 'db_port'; volara has no port setting, so set PGPORT in the environment")
    refuse_unknown(f"{path}: [{label}.db]", db, {"db_file"} if "db_file" in db else POSTGRES)


def refuse_unknown(where, keys, valid, hint=""):
    """Refuse the first key not in valid, naming the nearest valid key or listing them."""
    for k in keys:
        if k not in valid:
            near = difflib.get_close_matches(k, valid, 1, 0.8)
            fix = f"did you mean {near[0]!r}?" if near else f"keys are {', '.join(sorted(valid))}{hint}"
            raise click.ClickException(f"{where} has no key {k!r}; {fix}")
