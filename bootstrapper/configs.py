import click
import toml
import json
import os
from shutil import copytree
from pprint import pprint
import requests
from tqdm import tqdm
import zipfile
from ast import literal_eval

from funlib.geometry import Roi
from funlib.persistence import open_ds

from .segment import DEFAULTS as SEG_DEFAULTS
from .styles import cli_echo, cli_prompt, cli_confirm


BS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
MODEL_DIR = os.path.join(BS_DIR, "models")
MODEL_SHORT_NAMES = {
    "3d_affs_from_2d_affs": "3Af2A",
    "3d_affs_from_2d_lsd": "3Af2L",
    "3d_affs_from_2d_mtlsd": "3Af2M",
    "3d_affs_from_3d_lsd": "3Af3L",
}
MODEL_URLS = {
    "3d_affs_from_2d_affs": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.2.0/3d_affs_from_2d_affs.zip",
    "3d_affs_from_2d_lsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.2.0/3d_affs_from_2d_lsd.zip",
    "3d_affs_from_2d_mtlsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.2.0/3d_affs_from_2d_mtlsd.zip",
    "3d_affs_from_3d_lsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.2.0/3d_affs_from_3d_lsd.zip",
}


def get_setup_name(setup_dir):
    setup_name = os.path.basename(setup_dir)
    if "_from_" in setup_name:
        return MODEL_SHORT_NAMES[setup_name]
    else:
        return setup_name


def check_and_update(configs, style=None):
    click.echo()
    cli_echo(pprint(configs))

    if cli_confirm("Edit above?", style, default=False):
        if edited_configs := click.edit(toml.dumps(configs)):
            configs = toml.loads(edited_configs)
    return configs


def save_config(config, filename, style=None):
    with open(filename, "w") as f:
        toml.dump(config, f)
    cli_echo(f"{filename} saved successfully.", style, "success")


def copy_model_scripts(model_name, setup_dir, style="train"):
    src = os.path.abspath(os.path.join(BS_DIR, "models", model_name))
    cli_echo(f"Copying {src} to {setup_dir}..", style)
    copytree(src, setup_dir, dirs_exist_ok=True)

    # edit net config ?
    net_config_path = os.path.join(setup_dir, "net_config.json")
    if cli_confirm(f"Edit {net_config_path}?", style, default=False):
        click.edit(filename=net_config_path)


def get_sub_roi(in_array, offset=None, shape=None, style=None):
    """Get desired ROI within volume."""
    in_array = open_ds(in_array)
    full_roi = in_array.roi
    voxel_size = in_array.voxel_size
    full_shape = [s // v for s, v in zip(in_array.roi.shape, voxel_size)]

    if offset is None:
        offset = cli_prompt(
            f"Enter voxel offset as space-separated integers in {in_array.axis_names}",
            style,
            default="0 0 0",
        )
        offset = tuple(map(int, offset.strip().split()))

    if shape is None:
        shape = cli_prompt(
            f"Enter required voxel shape starting from {offset} as space-separated integers in {in_array.axis_names} (skipping will get remaining available shape)",
            style,
            default="0 0 0",
        )
        shape = tuple(map(int, shape.strip().split())) if shape != "0 0 0" else None

    roi_offset = [o * v for o, v in zip(offset, voxel_size)]

    if shape is None and roi_offset == [0, 0, 0]:
        roi_shape = [s * v for s, v in zip(full_shape, voxel_size)]
    else:
        remaining_shape = [
            fs - (ro // vs) for fs, ro, vs in zip(full_shape, roi_offset, voxel_size)
        ]
        if shape is None:
            roi_shape = [rem * vs for rem, vs in zip(remaining_shape, voxel_size)]
        else:
            roi_shape = [s * v for s, v in zip(shape, voxel_size)]
            roi_shape = [
                min(rs, rem * vs)
                for rs, rem, vs in zip(roi_shape, remaining_shape, voxel_size)
            ]

    roi = Roi(roi_offset, roi_shape)
    if not full_roi.contains(roi):
        roi = roi.intersect(full_roi)
        cli_echo(
            "ROI is not contained within the full volume's ROI. Cropping to {roi}..",
            style,
            "warning",
        )

    return roi.offset, roi.shape, voxel_size


def get_rag_db_config(sqlite_path=None, style="segment"):
    nodes_table = cli_prompt("Enter RAG nodes table name", style, default="nodes")
    edges_table = cli_prompt("Enter RAG edges table name", style, default="edges")

    if sqlite_path:
        db_file = cli_prompt(
            f"Enter SQLite RAG database file", style, default=sqlite_path
        )
        return {
            "db_file": db_file,
            "nodes_table": nodes_table,
            "edges_table": edges_table,
        }

    else:
        db_host = os.environ.get("RAG_DB_HOST")
        db_user = os.environ.get("RAG_DB_USER")
        db_password = os.environ.get("RAG_DB_PASSWORD")
        db_port = os.environ.get("RAG_DB_PORT")

        if not all([db_host, db_user, db_password, db_port]):
            cli_echo(
                "PgSQL Database credentials not found in environment variables..", style
            )
            db_host = cli_prompt("Enter PgSQL RAG database host", style)
            db_user = cli_prompt("Enter PgSQL RAG database user", style)
            db_password = cli_prompt(
                "Enter PgSQL RAG database password (input is hidden)",
                style,
                hide_input=True,
            )
            db_port = cli_prompt("Enter PgSQL RAG database port", style, type=int)

        db_name = cli_prompt("Enter PgSQL RAG database name", style)
        # write to env
        os.environ["RAG_DB_HOST"] = db_host
        os.environ["RAG_DB_USER"] = db_user
        os.environ["RAG_DB_PASSWORD"] = db_password
        os.environ["RAG_DB_PORT"] = str(db_port)

        return {
            "db_host": db_host,
            "db_user": db_user,
            "db_password": db_password,
            "db_port": db_port,
            "db_name": db_name,
            "nodes_table": nodes_table,
            "edges_table": edges_table,
        }


def choose_models(style="train"):
    model_names = []

    # models that take raw image as input
    image_models = sorted(
        [
            d
            for d in os.listdir(MODEL_DIR)
            if os.path.isdir(os.path.join(MODEL_DIR, d)) and "_from_" not in d
        ]
    )

    # models that take output from another model as input
    pred_models = sorted(
        [
            d
            for d in os.listdir(MODEL_DIR)
            if os.path.isdir(os.path.join(MODEL_DIR, d)) and "_from_" in d
        ]
    )

    # get first model
    i = 0
    previous_model = cli_prompt(
        f"Enter model name", style, type=click.Choice(image_models), show_choices=True
    )
    model_names.append(previous_model)

    while True:
        # check if a pred_model exists that can take prev model's output(s)
        compatible_pred_models = [
            m
            for m in pred_models
            if m.split("_from_")[1] in previous_model.split("_from_")[0]
        ]

        if not compatible_pred_models:
            break

        if len(compatible_pred_models) == 1:
            pred_model = compatible_pred_models[0]
        else:
            pred_model = cli_prompt(
                f"Enter model {i+2} name",
                style,
                type=click.Choice(compatible_pred_models),
                show_choices=True,
            )

        if cli_confirm(f"Enter whether to add {pred_model} to training config?", style, default=True):
            model_names.append(pred_model)
            previous_model = pred_model
            i += 1

    return model_names


def setup_models(model_names, parent_dir=None, style="train"):
    setup_dirs = []  # corresponding setup dirs for each model
    setups_to_train = []  # list of tuples (model_name, setup_dir)

    if parent_dir is None:
        parent_dir = os.getcwd()

    # get max setup number from existing setup dirs in parent_dir
    setup_num = (
        max(
            [
                int(d.split("_")[-1])
                for d in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, d))
                and os.path.exists(os.path.join(parent_dir, d, "net_config.json"))
            ],
            default=0,
        )
        + 1
    )

    # get setup dirs for each model
    for i, model_name in enumerate(model_names):
        if i == 0:
            setup_dir = cli_prompt(
                f"Enter setup dir for {model_name}",
                style,
                default=os.path.join(parent_dir, f"setup_{str(setup_num).zfill(2)}"),
                type=click.Path(),
            )
            setup_dir = os.path.abspath(setup_dir)
            copy_model_scripts(model_name, setup_dir)
            setups_to_train.append((model_name, setup_dir))
        else:
            choice = cli_prompt(
                f"Enter whether to use pretrained {model_name} or train from scratch?",
                style,
                type=click.Choice(["pretrained", "new"]),
                default="pretrained",
                show_choices=True,
            )
            if choice == "new":
                setup_dir = cli_prompt(
                    f"Enter new setup dir for {model_name}",
                    style,
                    default=os.path.join(
                        parent_dir, f"setup_{str(setup_num).zfill(2)}"
                    ),
                    type=click.Path(),
                )
                setup_dir = os.path.abspath(setup_dir)
                copy_model_scripts(model_name, setup_dir)
                setups_to_train.append((model_name, setup_dir))
            elif choice == "pretrained":
                setup_dir = os.path.join(MODEL_DIR, model_name)
                setup_dir = cli_prompt(
                    f"Enter existing setup dir for {model_name}",
                    style,
                    default=setup_dir,
                    type=click.Path(exists=True, file_okay=False, dir_okay=True),
                    show_default=True,
                )
                setup_dir = os.path.abspath(setup_dir)

                # check if pretrained model checkpoints exist
                checkpoints = [
                    c for c in os.listdir(setup_dir) if "model_checkpoint_" in c
                ]

                if not checkpoints:
                    cli_echo(f"No pretrained checkpoints found in {setup_dir}", style)

                    download = cli_confirm(
                        f"Enter whether to download pretrained checkpoints for {model_name}?", 
                        style,
                        default=True,
                    )

                    if download:
                        download_checkpoints(model_name, setup_dir)
                    else:
                        raise ValueError(
                            f"Please either download checkpoints or train from scratch"
                        )

        setup_dirs.append(setup_dir)
        setup_num += 1
        click.echo()

    return setup_dirs, setups_to_train


def choose_seg_method_params(aff_neighborhood=None, style="segment"):
    # choose method and params ?
    method = cli_prompt(
        "Enter segmentation method",
        style,
        type=click.Choice(["ws", "mws", "cc"]),
        show_choices=True,
        default="ws",
    )

    params = SEG_DEFAULTS[method]

    # update neighborhod if specified
    if aff_neighborhood is not None and method == "mws":
        params["aff_neighborhood"] = aff_neighborhood

    params = check_and_update(params, style)

    return method, params


def download_checkpoints(model_name, setup_dir, style="prepare"):
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}")

    url = MODEL_URLS[model_name]
    file_path = os.path.join(setup_dir, "checkpoints.zip")

    cli_echo(f"Downloading {model_name} checkpoints zip to {setup_dir}...", style)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(file_path, "wb") as file, tqdm(
        desc=model_name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

    # unzip checkpoints
    cli_echo(f"Unzipping {model_name} checkpoints in {setup_dir}...", style)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(setup_dir)

    # clean up
    os.remove(file_path)


def create_training_config(volumes, parent_dir=None, style="train"):

    click.echo()
    cli_echo(f"Creating training configs..", style)

    # get model names and setup dirs
    model_names = (
        choose_models()
    )  # sequence of model names, with output of one being input to the next
    setup_dirs, setups_to_train = setup_models(model_names, parent_dir)

    # get voxel size from volumes, assume all volumes have same voxel size
    voxel_size = volumes[list(volumes)[0]]["voxel_size"]
    configs = {}

    # create training configs
    for model_name, setup_dir in setups_to_train:
        max_iterations = cli_prompt(
            f"Enter max iterations for {model_name}", style, default=30001, type=int
        )
        save_checkpoints_every = cli_prompt(
            f"Enter save checkpoints every for {model_name}",
            style,
            default=5000,
            type=int,
        )
        save_snapshots_every = cli_prompt(
            f"Enter save snapshots every for {model_name}",
            style,
            default=1000,
            type=int,
        )

        train_config = {
            "setup_dir": setup_dir,
            "voxel_size": voxel_size,
            "max_iterations": max_iterations,
            "save_checkpoints_every": save_checkpoints_every,
            "save_snapshots_every": save_snapshots_every,
        }

        if "_from_" not in model_name:
            train_config["samples"] = [
                {
                    "raw": v["raw_dataset"],
                    "labels": v["labels_dataset"],
                    "mask": (
                        None
                        if "labels_mask_dataset" not in v
                        else v["labels_mask_dataset"]
                    ),
                }
                for _, v in volumes.items()
                if v["labels_dataset"] is not None
            ]

        configs[setup_dir] = check_and_update(train_config, style=style)

    return {"setup_dirs": setup_dirs, "configs": configs}


def create_prediction_configs(volumes, setup_dirs, style="predict"):

    click.echo()
    cli_echo(f"Creating prediction configs for {" -> ".join(setup_dirs)}", style)

    # get prediction iterations and setup names
    iterations = []
    setup_names = []
    for i, setup_dir in enumerate(setup_dirs):
        iteration = cli_prompt(
            f"Enter checkpoint iteration for model {i+1}: {os.path.basename(setup_dir)}",
            style,
            type=int,
            default=5000 * len(volumes) if i == 0 else 3000,
            show_default=True,
        )
        iterations.append(iteration)
        setup_names.append(get_setup_name(setup_dir))

    num_gpus = cli_prompt(
        "Enter number of GPUs to use for prediction", style, type=int, default=1
    )
    num_workers = cli_prompt(
        "Enter number of CPU workers to use for prediction",
        style,
        type=int,
        default=num_gpus,
    )

    # loop over volumes
    configs = {}
    for volume_name in volumes:
        pred_config = {}
        volume = volumes[volume_name]
        container = volume["output_container"]
        raw_array = volume["raw_dataset"]

        click.echo()
        cli_echo(f"Creating prediction configs for {volume_name}", style)

        roi_offset, roi_shape, _ = get_sub_roi(in_array=raw_array, style=style)
        output_datasets = []  # list of lists of output datasets per setup per volume

        # loop over setups
        for i, setup_dir in enumerate(setup_dirs):
            iteration = iterations[i]
            setup_name = setup_names[i]

            # get chain str
            chain = [f"{sn}_{it}" for sn, it in zip(setup_names[:i], iterations[:i])]
            chain_str = "--from--".join(chain)

            # get model outputs
            with open(os.path.join(setup_dir, "net_config.json"), "r") as f:
                model_outputs = json.load(f)["outputs"]

            # get in and out dataset names
            out_ds_prefix = f"{setup_name}"

            if i == 0 and chain_str == "":
                in_ds = [raw_array]
                out_ds = {
                    os.path.join(out_ds_prefix, str(iteration), x): model_outputs[x]
                    for x in model_outputs
                }
            else:
                in_ds = [os.path.join(container, ds) for ds in output_datasets[-1]]
                out_ds = {
                    os.path.join(out_ds_prefix, f"{iteration}--from--{chain_str}", x): model_outputs[x]
                    for x in model_outputs
                }

            output_datasets.append(out_ds)

            pred_config[f"{str(i+1).zfill(2)}-{setup_name}"] = {
                "setup_dir": setup_dir,
                "input_datasets": in_ds,
                "roi_offset": list(roi_offset),
                "roi_shape": list(roi_shape),
                "checkpoint": os.path.join(setup_dir, f"model_checkpoint_{iteration}"),
                "output_datasets_prefix": os.path.join(container, out_ds_prefix),
                "chain_str": chain_str,
                "num_workers": num_workers,
                "num_gpus": num_gpus,
            }

        configs[volume_name] = check_and_update(pred_config, style)

    pprint(output_datasets)
    out_affs_ds = [
        ds
        for x in output_datasets
        for ds in x
        if ds.split("/")[-1].startswith("3d_affs")
    ][-1]

    return {
        "out_affs_dataset": out_affs_ds,  # final 3d affs dataset to segment
        "out_pred_datasets": {
            ds: x[ds] for x in output_datasets for ds in x
        },  # flattened dict of pred datasets, assuming prefixes are unique for setups
        "configs": configs,
    }


def create_segmentation_configs(
    volumes, out_affs_ds, aff_neighborhood=None, style="segment"
):

    click.echo()
    cli_echo(f"Creating segmentation configs for {out_affs_ds}", style)

    method, params = choose_seg_method_params(aff_neighborhood)

    output_prefix = os.path.dirname(out_affs_ds)
    out_frags_ds = os.path.join(output_prefix, f"fragments_{method}")
    out_lut_dir = os.path.join(output_prefix, f"luts_{method}")
    out_seg_prefix = os.path.join(output_prefix, f"segmentations_{method}")

    configs = {}
    for volume_name in volumes:
        volume = volumes[volume_name]
        container = volume["output_container"]
        affs_array = os.path.join(container, out_affs_ds)
        frags_array = os.path.join(container, out_frags_ds)
        lut_dir = os.path.join(container, out_lut_dir)

        click.echo()
        cli_echo(f"Segmentation config for {volume_name}", style)

        # TODO: find way to get roi from predictions
        # roi_offset, roi_shape, voxel_size = get_roi(in_array=affs_array)

        do_blockwise = False

        if cli_confirm(f"Do blockwise = {do_blockwise}. Switch?", style, default=False):
            do_blockwise = not do_blockwise

        if do_blockwise and cli_confirm(
            f"Set block shape and context?", style, default=False
        ):
            block_shape = cli_prompt(
                "Enter block shape in voxels (e.g. 128,128,128), or 'roi' for single block with daisy",
                style,
            )
            context = cli_prompt("Enter context in voxels (e.g. 128,128,128)", style)
            if block_shape is not None and block_shape != "roi":
                block_shape = [int(x) for x in block_shape.split(",")]
            if context:
                context = [int(x) for x in context.split(",")]
            num_workers = cli_prompt(
                "Enter number of workers", style, default=10, type=int
            )
        else:
            block_shape = None
            context = None
            num_workers = 1

        # are raw masks available ?
        if "raw_mask_dataset" in volume and volume["raw_mask_dataset"] is not None:
            mask_dataset = volumes["raw_mask_dataset"]
        else:
            mask_dataset = None

        seg_config = {
            "affs_dataset": affs_array,
            "fragments_dataset": frags_array,
            "lut_dir": lut_dir,
            "seg_dataset_prefix": os.path.join(container, out_seg_prefix),
            "mask_dataset": mask_dataset,
            # "roi_offset": roi_offset,
            # "roi_shape": roi_shape,
            "block_shape": block_shape,
            "context": block_shape,
            "blockwise": do_blockwise,
            "num_workers": num_workers,
            f"{method}_params": params,
        }

        # get RAG db config
        if do_blockwise:
            sqlite_path = os.path.join(container, output_prefix, f"rag_{method}.db")

            # SQLite or not ?
            use_sqlite = not do_blockwise
            if cli_confirm(
                f"Use SQLite for RAG = {use_sqlite}. Switch?",
                style,
                default=False,
                show_default=True,
            ):
                use_sqlite = not use_sqlite

            sqlite_path = sqlite_path if use_sqlite else None

            # get rag db config
            seg_config["db"] = get_rag_db_config(sqlite_path)

        configs[volume_name] = check_and_update(seg_config, style)

    return {"out_seg_prefix": out_seg_prefix, "configs": configs}


def create_evaluation_configs(volumes, out_seg_prefix, pred_datasets, style="evaluate"):

    click.echo()
    cli_echo(f"Creating evaluation configs..", style)

    output_prefix = os.path.dirname(out_seg_prefix)

    configs = {}
    for volume_name in volumes:
        volume = volumes[volume_name]
        container = volume["output_container"]

        click.echo()
        cli_echo(f"Creating evaluation config for {out_seg_prefix}", style)

        # gt labels evaluation ?
        if cli_confirm(
            f"Are ground truth labels available for {volume_name}?",
            style,
            default=False,
        ):
            gt_labels_ds = os.path.abspath(
                cli_prompt(
                    "Enter path to ground truth labels dataset (press enter to skip)",
                    style,
                    type=click.Path(exists=True, dir_okay=True, file_okay=False),
                    default=None,
                )
            )
        else:
            gt_labels_ds = None

        # gt skeletons evaluation ?
        if cli_confirm(
            f"Are ground truth skeletons available for {volume_name}?",
            style,
            default=False,
        ):
            gt_skeletons_file = os.path.abspath(
                cli_prompt(
                    "Enter path to ground truth skeletons file (.graphml format) (press enter to skip)",
                    style,
                    type=click.Path(exists=True, dir_okay=False, file_okay=True),
                    default=None,
                )
            )
        else:
            gt_skeletons_file = None

        # self pred evaluation ?
        if cli_confirm(
            f"Compute prediction errors for {volume_name}?", style, default=True
        ):
            pred_choices = [
                ds for ds in pred_datasets if ds.split("/")[-1].startswith("3d_")
            ]

            if len(pred_choices) == 1:
                pred_ds_name = pred_choices[0]
            elif len(pred_choices) > 1:
                pred_ds_name = cli_prompt(
                    f"Enter {volume_name} prediction dataset to self-evaluate with",
                    style,
                    type=click.Choice(pred_choices),
                    default=pred_choices[-1],
                    show_choices=True,
                )
            else:
                pred_ds_name = None

            pred_type = pred_ds_name.split("/")[-1][3:7]
            try:
                pred_ds = pred_datasets[pred_ds_name]
            except:
                pred_ds = {}

            # check pred type, params
            if pred_type == "lsds":
                if "sigma" not in pred_ds:
                    pred_ds["sigma"] = literal_eval(
                        cli_prompt(
                            f"Enter sigma (in world units, as int or tuple of int) for {pred_ds_name}",
                            style,
                            default=str(volume["voxel_size"][-1] * 10),
                        )
                    )
            elif pred_type == "affs":
                if "neighborhood" not in pred_ds:
                    default_nbhd_str = "[[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [0, 8, 0], [0, 0, 8]]"
                    pred_ds["neighborhood"] = literal_eval(
                        cli_prompt(
                            f"Enter literal string of list of offsets to compute affinities from segmentation",
                            style,
                            default=default_nbhd_str,
                        )
                    )
                else:
                    if not isinstance(pred_ds["neighborhood"], list) or not all(
                        isinstance(x, list) and all(isinstance(y, int) for y in x)
                        for x in pred_ds["neighborhood"]
                    ):
                        raise ValueError(
                            f"{pred_ds_name}'s neighborhood must be a list of lists of int, not {pred_ds['neighborhood']}"
                        )
            else:
                raise ValueError(f"Unknown prediction type for {pred_ds_name}")
        else:
            pred_ds = None
            pred_ds_name = None

        # are raw masks available ?
        if "raw_mask_dataset" in volume and volume["raw_mask_dataset"] is not None:
            mask_dataset = volume["raw_mask_dataset"]
        else:
            mask_dataset = None

        # get evaluation ROI TODO: get from segmentation config
        # roi_offset, roi_shape, voxel_size = get_roi(in_array=out_segs)

        eval_config = {
            "out_result_dir": os.path.join(container, output_prefix),
            "seg_datasets_prefix": os.path.join(container, out_seg_prefix),
            "mask_dataset": mask_dataset,
            # "roi_offset": roi_offset,
            # "roi_shape": roi_shape,
        }

        if pred_ds is not None:
            eval_config["pred"] = {
                "pred_dataset": os.path.join(container, pred_ds_name),
                "thresholds": [0.1, 1.0],
            }
            if pred_type == "lsds":
                eval_config["pred"]["params"] = {"lsd_sigma": pred_ds["sigma"]}
            else:
                eval_config["pred"]["params"] = {
                    "aff_neighborhood": pred_ds["neighborhood"]
                }

        if gt_labels_ds is not None or gt_skeletons_file is not None:
            eval_config["gt"] = {
                "labels_dataset": gt_labels_ds,
                "skeletons_file": gt_skeletons_file,
            }

        configs[volume_name] = check_and_update(eval_config, style)

    return {"out_eval_dir": output_prefix, "configs": configs}


def create_filter_configs(volumes, in_seg_prefix, eval_dir, style="filter"):

    click.echo()
    cli_echo(f"Creating filter configs..", style)

    out_seg_ds_prefix = in_seg_prefix.replace("/segmentations_", "/pseudo_gt_ids_")
    out_mask_ds_prefix = in_seg_prefix.replace("/segmentations_", "/pseudo_gt_mask_")

    out_volumes = {}

    configs = {}
    for volume_name in volumes:
        volume = volumes[volume_name]
        container = volume["output_container"]
        out_seg_ds = os.path.join(container, out_seg_ds_prefix)
        out_mask_ds = os.path.join(container, out_mask_ds_prefix)

        # get filter ROI TODO: get from eval config
        # roi_offset, roi_shape, _ = get_roi(in_array=out_segs)

        filter_config = {
            "seg_datasets_prefix": os.path.join(container, in_seg_prefix),
            "eval_dir": os.path.join(container, eval_dir),
            "out_seg_dataset_prefix": out_seg_ds,
            "out_mask_dataset_prefix": out_mask_ds,
            # "roi_offset": roi_offset,
            # "roi_shape": roi_shape,
            "dust_filter": 500,
            "remove_outliers": True,
            "remove_z_fragments": 10,
            "overlap_filter": 0.0,
            "erode_out_mask": False,
        }

        configs[volume_name] = check_and_update(filter_config, style)

        # volumes for the next round
        out_volumes[volume_name] = {
            "name": volume_name,
            "raw_dataset": volume["raw_dataset"],
            "raw_mask_dataset": None if "raw_mask_dataset" not in volume else volume["raw_mask_dataset"],
            "labels_dataset": out_seg_ds,
            "labels_mask_dataset": out_mask_ds,
            "voxel_size": volume["voxel_size"],
            "previous_labels_datasets": [
                None if "labels_dataset" not in volume else volume["labels_dataset"],
            ]
            + volume.get("previous_labels_datasets", []),
            "previous_labels_mask_datasets": [
                None if "labels_mask_dataset" not in volume else volume["labels_mask_dataset"],
            ]
            + volume.get("previous_labels_mask_datasets", []),
        }

    return {"out_volumes": out_volumes, "configs": configs}


def make_round_configs(volumes, round_dir):
    """Create all configs for a model with given volumes."""

    run_dir = cli_prompt("Enter run directory", default=os.path.join(round_dir, "run"))
    os.makedirs(run_dir, exist_ok=True)

    train_config = create_training_config(volumes, round_dir)
    for i, setup_dir in enumerate(train_config["configs"]):
        save_config(
            train_config["configs"][setup_dir],
            os.path.join(run_dir, f"01_train_{str(i).zfill(2)}.toml"),
            style="train",
        )

    setup_dirs = train_config["setup_dirs"]
    pred_config = create_prediction_configs(volumes, setup_dirs)
    for volume_name in pred_config["configs"]:
        save_config(
            pred_config["configs"][volume_name],
            os.path.join(run_dir, f"02_pred_{volume_name}.toml"),
            style="predict",
        )

    out_affs_ds = pred_config["out_affs_dataset"]
    out_pred_datasets = pred_config["out_pred_datasets"]
    out_aff_neighborhood = out_pred_datasets[out_affs_ds]["neighborhood"]
    seg_configs = create_segmentation_configs(
        volumes, out_affs_ds, aff_neighborhood=out_aff_neighborhood
    )
    for volume_name in pred_config["configs"]:
        save_config(
            seg_configs["configs"][volume_name],
            os.path.join(run_dir, f"03_seg_{volume_name}.toml"),
            style="segment",
        )

    out_seg_prefix = seg_configs["out_seg_prefix"]
    eval_configs = create_evaluation_configs(
        volumes,
        out_seg_prefix,
        out_pred_datasets,
    )
    for volume_name in pred_config["configs"]:
        save_config(
            eval_configs["configs"][volume_name],
            os.path.join(run_dir, f"04_eval_{volume_name}.toml"),
            style="evaluate",
        )

    out_eval_dir = eval_configs["out_eval_dir"]
    filter_configs = create_filter_configs(volumes, out_seg_prefix, out_eval_dir)
    for volume_name in pred_config["configs"]:
        save_config(
            filter_configs["configs"][volume_name],
            os.path.join(run_dir, f"05_filter_{volume_name}.toml"),
            style="filter",
        )

    out_volumes = filter_configs["out_volumes"]
    return out_volumes
