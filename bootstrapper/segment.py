import click
import logging
from pprint import pprint
import toml
from ast import literal_eval

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULTS = {
    "ws": {
        "fragments_in_xy": True,
        "min_seed_distance": 10,
        "epsilon_agglomerate": 0.0,
        "filter_fragments": 0.05,
        "replace_sections": None,
        "thresholds_minmax": [0, 1],
        "thresholds_step": 0.05,
        "thresholds": [0.2, 0.35, 0.5],
        "merge_function": "mean",
        "sigma": None,
        "noise_eps": None,
        "bias": None,
    },
    "mws": {
        "aff_neighborhood": [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [-2, 0, 0],
            [0, -8, 0],
            [0, 0, -8],
        ],
        "bias": [-0.2, -0.2, -0.2, -0.8, -0.8, -0.8],
        "sigma": None,
        "noise_eps": 0.001,
        "strides": [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        "randomized_strides": True,
        "filter_fragments": 0.05,
        "remove_debris": 500,
    },
    "cc": {
        "threshold": 0.5,
        "sigma": None,
        "noise_eps": None,
        "remove_debris": 0,
    },
}


def parse_params(param_str):
    try:
        return literal_eval(param_str)
    except:
        return param_str


def get_method_params(method, params):
    ret = {}

    for p_str in params:
        p, v = p_str.split("=")
        if p in DEFAULTS[method]:
            ret[p] = parse_params(v)
        else:
            raise ValueError(f"Invalid {method} parameter {p}")

    return ret


def get_seg_config(config_file, method, **kwargs):
    # load config
    with open(config_file, "r") as f:
        config = toml.load(f)

    # override config values with provided kwargs, except method specific params
    for key, value in kwargs.items():
        if key != "param" and value is not None:
            config[key] = value

    # override method specific defaults with provided params
    if "param" in kwargs:
        params = config.get(f"{method}_params", {}) | get_method_params(
            method, kwargs["param"]
        )
    else:
        params = DEFAULTS[method] | config.get(f"{method}_params", {})

    # delete config param dicts
    for x in config.copy():
        if x.endswith("_params"):
            del config[x]

    # check blockwise -- check if db info is provided
    if config.get("blockwise", False):
        if "db" not in config:
            raise ValueError("Blockwise requires a database config!")

        if "lut_dir" not in config:
            config["lut_dir"] = config["seg_dataset_prefix"].replace(
                "segmentations", "luts"
            )

    return config | params


def run_segmentation(config_file, mode="ws", **kwargs):
    config = get_seg_config(config_file, mode, **kwargs)
    pprint(config)

    if mode == "ws":
        from .post.watershed import watershed_segmentation

        watershed_segmentation(config)
    elif mode == "mws":
        from .post.watershed_mutex import mutex_watershed_segmentation

        mutex_watershed_segmentation(config)
    elif mode == "cc":
        from .post.connected_components import cc_segmentation

        cc_segmentation(config)
    else:
        raise ValueError(f"Unknown segmentation mode: {mode}")


@click.command()
@click.argument(
    "config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option("--ws", "-ws", is_flag=True, help="Watershed segmentation (waterz)")
@click.option("--mws", "-mws", is_flag=True, help="Mutex watershed segmentation")
@click.option("--cc", "-cc", is_flag=True, help="Connected componenents segmentation")
@click.option(
    "--roi-offset",
    "-ro",
    type=str,
    help="Offset of ROI in world units (space separated integers)",
)
@click.option(
    "--roi-shape",
    "-rs",
    type=str,
    help="Shape of ROI in world units (space separated integers)",
)
@click.option(
    "--blockwise", "-b", is_flag=True, default=None, help="Run blockwise segmentation, with daisy"
)
@click.option(
    "--num-workers",
    "-n",
    type=int,
    help="Number of workers, for blockwise segmentation",
)
@click.option(
    "--block-shape",
    "-bs",
    type=str,
    help="Block shape, for blockwise segmentation (space separated integers or 'roi')",
)
@click.option(
    "--block-context",
    "-bc",
    type=str,
    help="Block context, for blockwise segmentation (space separated integers)",
)
@click.option(
    "--param",
    "-p",
    multiple=True,
    help="Method specific parameters to override in config (e.g. -p 'thresholds=[0.2,0.3]')",
)
def segment(config_file, ws, mws, cc, **kwargs):
    """
    Segment affinities as specified in config_file.
    """
    methods = []

    with open(config_file, "r") as f:
        config = toml.load(f)
        method_params = [
            config.get(f"{method}_params", None) for method in ["ws", "mws", "cc"]
        ]

    if any([ws, mws, cc]):
        if ws:
            methods.append("ws")
        if mws:
            methods.append("mws")
        if cc:
            methods.append("cc")
    elif any(method_params):
        methods = [
            method
            for method, params in zip(["ws", "mws", "cc"], method_params)
            if params
        ]
    else:
        methods = ["ws"]

    for method in methods:
        run_segmentation(config_file, method, **kwargs)
