import json

import numpy as np
import zarr

# param -> short code; dict order is the name order
CODES = {
    "merge_function": "mf",
    "threshold": "t",
    "global_bias": "gb",
    "fragments_in_xy": "xy",
    "min_seed_distance": "msd",
    "seed_eps": "seps",
    "epsilon_agglomerate": "ea",
    "sigma": "sig",
    "noise_eps": "eps",
    "bias": "b",
    "strides": "st",
    "randomized_strides": "rs",
    "filter_fragments": "ff",
    "remove_debris": "rd",
}


def fmt(value, sep="_"):
    """All-equal lists collapse to one value; nested lists join with '.'."""
    if isinstance(value, (list, tuple)):
        parts = [fmt(v, sep=".") for v in value]
        return parts[0] if len(set(parts)) == 1 else sep.join(parts)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_name(params):
    """Name from the params a path actually applies: every non-None param is
    included, in CODES order. Booleans render as bare code (True) or code0."""
    parts = []
    for key, code in CODES.items():
        if key not in params or params[key] is None:
            continue
        value = params[key]
        if isinstance(value, bool):
            parts.append(code if value else f"{code}0")
        else:
            parts.append(f"{code}{fmt(value)}")
    return "--".join(parts)


def _jsonable(value):
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def dump_params(store, params):
    """Record the resolved params on the output dataset's zarr attrs."""
    # r+ so a missing dataset fails loudly instead of creating a stray group
    zarr.open(store, mode="r+").attrs["bs_params"] = {
        k: _jsonable(v) for k, v in params.items()
    }


def dump_lut_params(lut_path, params):
    with open(f"{lut_path}.json", "w") as f:
        json.dump({k: _jsonable(v) for k, v in params.items()}, f, indent=2)
