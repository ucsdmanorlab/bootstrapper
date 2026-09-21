"""Config files are kind tables. [volumes.<name>] describes data; [train.<name>],
[predict.<name>], [segment], [segment.<name>], [refine] and [evaluate] are steps. A file
holds one step or a whole round; a step is named only when a kind has several."""
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
FLAT_HINTS = (("segment", {"affs_dataset"}), ("train", {"samples", "max_iterations"}),
              ("evaluate", {"seg_datasets_prefix", "seg_datasets"}), ("predict", {"checkpoint"}))


@dataclass
class Step:
    kind: str
    name: str
    keys: dict
    volumes: dict = field(default_factory=dict)  # per-volume keys, from [<step>.<volume>]

    @property
    def label(self):
        return self.kind if self.name == self.kind else f"{self.kind}.{self.name}"


@dataclass
class Config:
    path: str
    volumes: dict
    steps: list


def load(path):
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    volumes = doc.pop("volumes", {})
    doc.pop("round", None)
    doc.pop("run", None)

    flat = {k for k, v in doc.items() if not isinstance(v, dict)}
    if flat:
        kind = next((k for k, hint in FLAT_HINTS if hint & set(doc)), "<kind>")
        fix = f"put these keys under [{kind}]" if kind in ("segment", "evaluate") else f"put these keys under [{kind}.<name>]"
        old = OLD_PARAMS & set(doc)
        if old:
            fix += f" and rename [{old.pop()}] to [{kind}.params]"
        raise click.ClickException(f"{path}: {fix}")

    steps = []
    for kind, table in doc.items():
        if kind not in KINDS:
            hint = " (a prediction: put its keys under [predict.<name>])" if any(isinstance(v, dict) and "checkpoint" in v for v in table.values()) else ""
            raise click.ClickException(f"{path}: unknown table [{kind}]; tables are volumes, {', '.join(KINDS)}{hint}")
        if _is_step(kind, table, volumes):
            steps.append(_step(kind, kind, table, volumes, path))
        else:
            for name, sub in table.items():
                if name in volumes:
                    raise click.ClickException(f"{path}: [{kind}.{name}] is a step named like the volume {name!r}")
                steps.append(_step(kind, name, sub, volumes, path))
    return Config(path, volumes, steps)


def steps_of(path, kind, name=None):
    """The steps of one kind in a file, or the one named."""
    cfg = load(path)
    steps = [s for s in cfg.steps if s.kind == kind]
    if not steps:
        raise click.ClickException(f"{path}: no [{kind}] step")
    if name:
        steps = [s for s in steps if s.name == name]
        if not steps:
            names = ", ".join(s.name for s in cfg.steps if s.kind == kind)
            raise click.ClickException(f"{path}: no [{kind}.{name}]; {kind} steps are {names}")
    return cfg, steps


def runs(step):
    """One flat config per volume table, else one for the step itself."""
    if not step.volumes:
        return [(None, dict(step.keys))]
    return [(vol, step.keys | keys) for vol, keys in step.volumes.items()]


def _belongs(kind, name, volumes):
    return name in STEP_TABLES or name in OLD_PARAMS or name in volumes or (kind == "refine" and name in OPS)


def _is_step(kind, table, volumes):
    """A table with a scalar key is one step; so is one whose sub-tables all belong to a
    step. Otherwise its sub-tables are named steps."""
    if any(not isinstance(v, dict) for v in table.values()):
        return True
    return all(_belongs(kind, k, volumes) for k in table)


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
            ops = f"; ops are {', '.join(OPS)}" if kind == "refine" else ""
            raise click.ClickException(
                f"{path}: [{label}] has no key {unknown[0]!r}; keys are {', '.join(sorted(KEYS[kind]))}{ops}"
            )
    return Step(kind, name, keys, vols)
