"""Config files are kind tables. [volumes.<name>] describes data; [train.<name>],
[predict.<name>], [segment], [segment.<name>], [refine] and [evaluate] are steps. A file
holds one step or a whole round; a step is named only when a kind has several."""
import os
import tomllib
from dataclasses import dataclass, field

import click

KINDS = ("train", "predict", "segment", "refine", "evaluate")
STEP_TABLES = {"params", "db", "pred", "gt", "samples"}
OPS = ("size-filter", "outlier-filter", "z-filter", "remap", "fill-holes", "dilate",
       "erode", "opening", "closing", "mask")
OLD_PARAMS = {"ws_params", "mws_params", "cc_params"}
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

    if any(not isinstance(v, dict) for v in doc.values()):
        raise click.ClickException(
            f"{path}: keys at the top level; put them under [segment], [evaluate], [train.<name>] "
            "or [predict.<name>], and a [<method>_params] table under [segment.params]"
        )

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
        unknown = [k for k in keys if k not in KEYS[kind] and k not in STEP_TABLES and k != "ops"]
        if unknown:
            k = unknown[0]
            ops = f"; ops are {', '.join(OPS)}" if kind == "refine" else ""
            if isinstance(keys[k], dict):
                vols_known = f"volumes are {', '.join(volumes)}" if volumes else "no volumes are defined"
                raise click.ClickException(f"{path}: [{label}.{k}] is neither a table of [{label}] nor a volume; {vols_known}{ops}")
            raise click.ClickException(f"{path}: [{label}] has no key {k!r}; keys are {', '.join(sorted(KEYS[kind]))}{ops}")
    return Step(kind, name, keys, vols)
