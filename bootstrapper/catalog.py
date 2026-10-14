"""The models, what each reads and gives, and the chains that make one round."""
import json
import os
import re
from dataclasses import dataclass

import click

from .configs import MODEL_DIR, MODEL_NAMES, MODEL_URLS


@dataclass(frozen=True)
class Model:
    name: str
    short: str  # 3d_affs_from_2d_lsd is 3Af2L
    inputs: tuple
    outputs: tuple
    shipped: bool  # a pretrained checkpoint downloads; the others train on your labels

    @property
    def reads_raw(self):
        return self.inputs == ("raw",)


def _model(name):
    with open(os.path.join(MODEL_DIR, name, "net_config.json")) as f:
        net = json.load(f)
    short = re.sub(r"d_([a-z])[a-z]*", lambda m: m[1].upper(), name).replace("_from_", "f")
    return Model(name, short, tuple(net["inputs"]), tuple(net["outputs"]), name in MODEL_URLS)


MODELS = {name: _model(name) for name in MODEL_NAMES}
_BY_KEY = {key.lower(): m for m in MODELS.values() for key in (m.name, m.short)}


def download_fix(name, round_path=None):
    """The commands that download the checkpoint of the shipped model name."""
    again = f"bs prepare {round_path} again, or " if round_path else ""
    return f"run {again}bs utils download-ckpts -m {MODELS[name].short}"


def _list(models):
    return ", ".join(f"{m.name} ({m.short})" for m in models)


def _arrow(chain):
    return " → ".join(m.name for m in chain)


def _readers(model):
    return [m for m in MODELS.values() if not m.reads_raw and set(m.inputs) <= set(model.outputs)]


def parse_chain(text):
    """Models by name or short form, split by spaces, commas or arrows; case does not matter."""
    tokens = [t for t in re.split(r"[\s,]+|->|→", text) if t]
    if not tokens:
        raise click.ClickException(f"give one or more models, such as 2M 3Af2L; models: {_list(MODELS.values())}")
    for t in tokens:
        if t.lower() not in _BY_KEY:
            raise click.ClickException(f"unknown model {t!r}; models: {_list(MODELS.values())}")
    return [_BY_KEY[t.lower()] for t in tokens]


def check_round(chain):
    """Refuse a chain unless it starts from raw, gives each model what it reads, and ends in 3D affinities."""
    first = chain[0]
    if not first.reads_raw:
        makers = [m for m in MODELS.values() if set(first.inputs) <= set(m.outputs)]
        raise click.ClickException(
            f"{first.name} reads {' and '.join(first.inputs)}, not raw. Start with a model that makes them: {_list(makers)}")
    for i, (prev, model) in enumerate(zip(chain, chain[1:]), 1):
        if model.reads_raw and "3d_affs" in prev.outputs:
            check_round(chain[i:])
            raise click.ClickException(
                f"{model.name} trains on this round's segmentation, so it starts the next round. "
                f"This round: {_arrow(chain[:i])}; next round: {_arrow(chain[i:])}")
        if not set(model.inputs) <= set(prev.outputs):
            readers = _readers(prev)
            fix = f"Models that read {prev.name}: {_list(readers)}" if readers else f"End the round at {prev.name}"
            raise click.ClickException(
                f"{model.name} reads {' and '.join(model.inputs)}; {prev.name} gives {' and '.join(prev.outputs)}. {fix}")
    last = chain[-1]
    if "3d_affs" not in last.outputs:
        raise click.ClickException(
            f"{last.name} gives {' and '.join(last.outputs)}; a round ends with 3D affinities. "
            f"Add one after it: {_list(_readers(last))}")


def valid_chains():
    """Every chain that check_round accepts, in name order."""
    def grow(chain):
        if "3d_affs" in chain[-1].outputs:
            yield chain
        for m in _readers(chain[-1]):
            yield from grow(chain + [m])
    return [chain for m in MODELS.values() if m.reads_raw for chain in grow([m])]


def default_chain(labels_2d):
    """2D labels train 2d_mtlsd for the shipped 3d_affs_from_2d_mtlsd; 3D labels train 3d_mtlsd."""
    return parse_chain("2M 3Af2M" if labels_2d else "3M")


def describe(chain):
    """2d_mtlsd → 3d_affs_from_2d_lsd; 3d_affs_from_2d_lsd reads only the 2d_lsds"""
    notes = [f"; {m.name} reads only the {' and '.join(m.inputs)}"
             for prev, m in zip(chain, chain[1:]) if set(prev.outputs) - set(m.inputs)]
    return _arrow(chain) + "".join(notes)
