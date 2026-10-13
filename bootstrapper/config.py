"""Config files are kind tables. [volumes.<name>] describes data; [train.<name>],
[predict.<name>], [segment], [segment.<name>], [refine] and [evaluate] are steps. A file
holds one step or a whole round; a step is named only when a kind has several."""
import difflib
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
KEYS = {
    "predict": {"setup_dir", "checkpoint", "input_datasets", "output_datasets_prefix",
                "chain_str", "num_workers", "num_gpus", "roi_offset", "roi_shape"},
    "segment": {"method", "affs_dataset", "fragments_dataset", "lut_dir", "seg_dataset_prefix",
                "mask_dataset", "blockwise", "block_shape", "context", "num_workers",
                "roi_offset", "roi_shape"},
    "refine": {"in_array", "num_workers"},
    "evaluate": {"seg_datasets_prefix", "seg_datasets", "mask_dataset", "out_result", "out_dir",
                 "num_workers", "thresholds", "roi_offset", "roi_shape"},
}


@dataclass
class Step:
    kind: str
    name: str
    keys: dict
    volumes: dict = field(default_factory=dict)  # per-volume keys, from [<step>.<volume>]

    @property
    def label(self):
        return self.kind if self.name == self.kind else f"{self.kind}.{self.name}"


def load(path):
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    volumes = doc.pop("volumes", {})
    beside = os.path.join(os.path.dirname(os.path.abspath(path)), "volumes.toml")
    if not volumes and os.path.exists(beside) and os.path.abspath(path) != beside:
        with open(beside, "rb") as f:
            volumes = tomllib.load(f).get("volumes", {})
    doc.pop("round", None)
    doc.pop("run", None)

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
            steps.append(_step(kind, kind, table, volumes, path))
        else:
            for name, sub in table.items():
                if name in volumes:
                    raise click.ClickException(f"{path}: [{kind}.{name}] is a step named like the volume {name!r}")
                steps.append(_step(kind, name, sub, volumes, path))
    return steps


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
    """One flat config per volume table, else one for the step itself."""
    if not step.volumes:
        return [(None, dict(step.keys))]
    return [(vol, step.keys | keys) for vol, keys in step.volumes.items()]


def _own(kind, name):
    return name in STEP_TABLES or name in OLD_PARAMS or (kind == "refine" and name in OPS)


def _is_step(kind, table, volumes):
    """A table with a scalar key is one step; so is one whose sub-tables are all its own
    or volumes. Otherwise its sub-tables are named steps."""
    if any(not isinstance(v, dict) for v in table.values()):
        return True
    return all(_own(kind, k) or k in volumes for k in table)


def _step(kind, name, table, volumes, path):
    label = kind if name == kind else f"{kind}.{name}"
    keys = dict(table)
    old = OLD_PARAMS & set(keys)
    if old:
        raise click.ClickException(f"{path}: [{label}.{old.pop()}] is now [{label}.params]")
    vols = {k: keys.pop(k) for k in list(keys) if k in volumes}
    if kind == "refine":
        keys["ops"] = {k: keys.pop(k) for k in list(keys) if k in OPS}
    if kind != "train":
        _check(path, label, kind, keys, volumes)
        for vol, t in vols.items():
            _check(path, f"{label}.{vol}", kind, t, volumes)
    return Step(kind, name, keys, vols)


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
