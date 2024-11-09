import click
import yaml
import json
import os
from shutil import copytree
from pprint import pprint
import requests
from tqdm import tqdm
import zipfile

from funlib.geometry import Roi
from funlib.persistence import open_ds

from .segment import DEFAULTS as SEG_DEFAULTS


DEFAULT_PROMPT_STYLE = {"fg": "cyan"}
DEFAULT_INFO_STYLE = {"fg": "cyan", "bold": True}
DEFAULT_TRAIN_STYLE = {"fg": "green"}
DEFAULT_PRED_STYLE = {"fg": "yellow"}
DEFAULT_SEG_STYLE = {"fg": "red"}
DEFAULT_EVAL_STYLE = {"fg": "magenta"}
DEFAULT_FILTER_STYLE = {"fg": "blue"}

BS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
MODEL_DIR = os.path.join(BS_DIR, "models")
MODEL_SHORT_NAMES = {
    "3d_affs_from_2d_affs": "3Af2A",
    "3d_affs_from_2d_lsd": "3Af2L",
    "3d_affs_from_2d_mtlsd": "3Af2M",
    "3d_affs_from_3d_lsd": "3Af3L",
}
MODEL_URLS = {
    "3d_affs_from_2d_affs": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.1.1/3d_affs_from_2d_affs.zip",
    "3d_affs_from_2d_lsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.1.1/3d_affs_from_2d_lsd.zip",
    "3d_affs_from_2d_mtlsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.1.1/3d_affs_from_2d_mtlsd.zip",
    "3d_affs_from_3d_lsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.1.1/3d_affs_from_3d_lsd.zip",
}


def get_setup_name(setup_dir):
    setup_name = os.path.basename(setup_dir)
    if '_from_' in setup_name:
        return MODEL_SHORT_NAMES[setup_name]
    else:
        return setup_name


def check_and_update(configs, style=DEFAULT_PROMPT_STYLE):
    multiple = isinstance(configs, list)
    configs = [configs] if not multiple else configs

    click.echo()
    for config in configs:
        click.secho(pprint(config))

    if not click.confirm(
        click.style("Enter confirmation for values above (y/n)", **style), default=True
    ):
        if edited_yaml := click.edit(yaml.dump_all(configs)):
            configs = list(yaml.safe_load_all(edited_yaml))

    return configs[0] if not multiple else configs


def save_config(config, filename, style=DEFAULT_INFO_STYLE):
    with open(filename, "w") as f:
        yaml.dump(config, f)
    click.secho(f"{filename} saved successfully.", **style)


def copy_model_scripts(model_name, setup_dir):
    src = os.path.abspath(os.path.join(BS_DIR, "models", model_name))
    click.secho(f"Copying {src} to {setup_dir}", **DEFAULT_TRAIN_STYLE)
    copytree(src, setup_dir, dirs_exist_ok=True)


def get_sub_roi(in_array, offset=None, shape=None, style=DEFAULT_PROMPT_STYLE):
    """Get desired ROI within volume."""
    in_array = open_ds(in_array)
    full_roi = in_array.roi
    voxel_size = in_array.voxel_size
    full_shape = [s // v for s, v in zip(in_array.roi.shape, voxel_size)]

    if offset is None:
        offset = click.prompt(
            click.style(
                f"Enter voxel offset as space-separated integers in {in_array.axis_names}",
                **style,
            ),
            type=str,
            default="0 0 0",
            show_default=False,
        )
        offset = tuple(map(int, offset.strip().split()))

    if shape is None:
        shape = click.prompt(
            click.style(
                f"Enter required voxel shape starting from {offset} as space-separated integers in {in_array.axis_names} (skipping will get remaining available shape)",
                **style,
            ),
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
        click.secho(
            "ROI is not contained within the full volume's ROI. Cropping to..",
            **style,
        )
        roi = roi.intersect(full_roi)
        click.secho(f"{roi}", **style)

    return roi.offset, roi.shape, voxel_size


def get_rag_db_config(sqlite_path=None):
    nodes_table = click.prompt(
        click.style("Enter RAG nodes table name", **DEFAULT_SEG_STYLE), default="nodes", show_default=True
    )
    edges_table = click.prompt(
        click.style("Enter RAG edges table name", **DEFAULT_SEG_STYLE), default="edges", show_default=True
    )

    if sqlite_path:
        db_file = click.prompt(
            click.style(f"Enter SQLite RAG database file", **DEFAULT_SEG_STYLE), default=sqlite_path, show_default=True
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
            click.secho(
                "PgSQL Database credentials not found in environment variables.",
                **DEFAULT_SEG_STYLE,
            )
            db_host = click.prompt(click.style("Enter PgSQL RAG database host", **DEFAULT_SEG_STYLE))
            db_user = click.prompt(click.style("Enter PgSQL RAG database user", **DEFAULT_SEG_STYLE))
            db_password = click.prompt(
                click.style("Enter PgSQL RAG database password (input is hidden)", **DEFAULT_SEG_STYLE), hide_input=True
            )
            db_port = click.prompt(click.style("Enter PgSQL RAG database port", **DEFAULT_SEG_STYLE), type=int)

        db_name = click.prompt(click.style("Enter PgSQL RAG database name", **DEFAULT_SEG_STYLE))
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


def choose_models():
    
    model_names = []

    # models that take raw image as input
    image_models = sorted(
        [
            d for d in os.listdir(MODEL_DIR) 
            if os.path.isdir(os.path.join(MODEL_DIR, d))
            and "_from_" not in d
        ]
    )

    # models that take output from another model as input
    pred_models = sorted(
        [
            d for d in os.listdir(MODEL_DIR)
            if os.path.isdir(os.path.join(MODEL_DIR, d))
            and "_from_" in d
        ]
    )

    # get first model
    i = 0
    previous_model = click.prompt(
        click.style(f"Enter model name", **DEFAULT_TRAIN_STYLE),
        type=click.Choice(image_models),
        show_choices=True,
    )

    model_names.append(previous_model)

    while True:
        # check if a pred_model exists that can take prev model's output(s)
        compatible_pred_models = [
            m for m in pred_models if m.split('_from_')[1] in previous_model.split('_from_')[0]
        ]
        
        if not compatible_pred_models:
            break

        if len(compatible_pred_models) == 1:
            pred_model = compatible_pred_models[0]
        else:
            pred_model = click.prompt(
                click.style(f"Enter model {i+2} name", **DEFAULT_TRAIN_STYLE),
                type=click.Choice(compatible_pred_models),
                show_choices=True
            )

        if click.confirm(
            click.style(f"Add {pred_model} to training config?", **DEFAULT_TRAIN_STYLE),
            default=True,
        ): 
            model_names.append(pred_model)
            previous_model = pred_model
            i += 1

    return model_names


def choose_seg_method_params():
    # choose method and params ?
    if click.confirm(
        click.style("Specify segmentation method?", **DEFAULT_SEG_STYLE),
        default=False,
    ):
        # specify method ?
        method = click.prompt(
            click.style("Enter segmentation method", **DEFAULT_SEG_STYLE),
            type=click.Choice(["ws", "mws", "cc"]),
            show_choices=True,
            default="ws",
        )
    else:
        method = "ws"

    # specify params ?  
    if click.confirm(
        click.style(f"Specify {method} segmentation parameters?", **DEFAULT_SEG_STYLE),
        default=False,
    ):
        params = check_and_update(SEG_DEFAULTS[method])
    else:
        params = SEG_DEFAULTS[method]

    return method, params


def download_checkpoints(model_name, setup_dir):

    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}")
    
    url = MODEL_URLS[model_name]
    file_path = os.path.join(setup_dir, "checkpoints.zip")
    
    click.secho(f"Downloading {model_name} checkpoints zip to {setup_dir}...", **DEFAULT_INFO_STYLE)
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
    click.secho(f"Unzipping {model_name} checkpoints in {setup_dir}...", **DEFAULT_INFO_STYLE)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(setup_dir)

    # clean up
    os.remove(file_path)


def create_training_config(volumes, parent_dir=None):

    click.echo()
    click.secho(
        f"Creating training configs", **DEFAULT_TRAIN_STYLE
    )

    if parent_dir is None:
        parent_dir = os.getcwd()

    # get max setup number from existing setup dirs
    setup_num = max(
        [int(d.split('_')[-1]) for d in os.listdir(parent_dir)
         if os.path.isdir(os.path.join(parent_dir, d))
         and os.path.exists(os.path.join(parent_dir, d, 'net_config.json'))],
        default=0
    ) + 1

    # get model names and setup dirs
    model_names = choose_models() # sequence of model names, with output of one being input to the next
    setup_dirs = [] # corresponding setup dirs
    setups_to_train = [] # list of tuples (model_name, setup_dir)
    
    # get setup dirs for each model
    for i, model_name in enumerate(model_names):
        if i == 0:
            setup_dir = click.prompt(
                click.style(f"Enter setup dir for {model_name}", **DEFAULT_TRAIN_STYLE),
                default=os.path.join(parent_dir, f"setup_{str(setup_num).zfill(2)}"),
                type=click.Path(),
            )
            setup_dir = os.path.abspath(setup_dir)
            copy_model_scripts(model_name, setup_dir)

            setups_to_train.append((model_name, setup_dir))

        else:
            choice = click.prompt(
                click.style(f"Use pretrained {model_name} or train from scratch?", **DEFAULT_TRAIN_STYLE),
                type=click.Choice(["pretrained", "new"]),
                default="pretrained",
                show_choices=True,
            )
            if choice == "new":
                setup_dir = click.prompt(
                    click.style(f"Enter new setup dir for {model_name}", **DEFAULT_TRAIN_STYLE),
                    default=os.path.join(parent_dir, f"setup_{str(setup_num).zfill(2)}"),
                    type=click.Path(),
                )
                setup_dir = os.path.abspath(setup_dir)
                copy_model_scripts(model_name, setup_dir)
                setups_to_train.append((model_name, setup_dir))

            elif choice == "pretrained":
                setup_dir = os.path.join(os.path.dirname(__file__), "models", model_name)
                setup_dir = click.prompt(
                    click.style(f"Enter existing setup dir for {model_name}", **DEFAULT_TRAIN_STYLE),
                    default=setup_dir,
                    type=click.Path(exists=True, file_okay=False, dir_okay=True),
                    show_default=True,
                )
                setup_dir = os.path.abspath(setup_dir)

                # check if pretrained model checkpoints exist
                checkpoints = [
                    c for c in os.listdir(setup_dir) if 'model_checkpoint_' in c
                ]

                if not checkpoints:
                    click.secho(f"No pretrained checkpoints found in {setup_dir}", **DEFAULT_TRAIN_STYLE)

                    download = click.confirm(
                        click.style(f"Download pretrained checkpoints for {model_name}?", **DEFAULT_TRAIN_STYLE),
                        default=True,
                    )

                    if download:
                        download_checkpoints(model_name, setup_dir)
                    else:
                        raise ValueError(f"Please either download checkpoints or train from scratch")
                
        setup_dirs.append(setup_dir)
        setup_num += 1

    # create training configs
    # get voxel size from volumes, assume all volumes have same voxel size
    voxel_size = volumes[0]["voxel_size"]
    configs = {}

    for model_name, setup_dir in setups_to_train:

        max_iterations = click.prompt(click.style(f"Enter max iterations for {model_name}", **DEFAULT_TRAIN_STYLE), default=30001, type=int)
        save_checkpoints_every = click.prompt(
            click.style(f"Enter save checkpoints every for {model_name}", **DEFAULT_TRAIN_STYLE), default=5000, type=int
        )
        save_snapshots_every = click.prompt(
            click.style(f"Enter save snapshots every for {model_name}", **DEFAULT_TRAIN_STYLE), default=1000, type=int
        )

        train_config = {
            "setup_dir": setup_dir,
            "voxel_size": voxel_size,
            "max_iterations": max_iterations,
            "save_checkpoints_every": save_checkpoints_every,
            "save_snapshots_every": save_snapshots_every,
        }

        if "lsd" in model_name:
            train_config["sigma"] = click.prompt(
                click.style("Enter sigma for LSD model", **DEFAULT_TRAIN_STYLE), default=10 * voxel_size[-1], type=int
            )

        if '_from_' not in model_name:
            train_config['samples'] = [
                {
                    "raw": v["raw_dataset"],
                    "labels": v["labels_dataset"],
                    "mask": v["labels_mask_dataset"],
                }
                for v in volumes
                if v["labels_dataset"] is not None
            ]

        configs[setup_dir] = check_and_update(train_config, style=DEFAULT_TRAIN_STYLE)

    return {
        'setup_dirs': setup_dirs,
        'configs': configs
    }


def create_prediction_configs(volumes, setup_dirs):

    click.echo()
    click.secho(
        f"Prediction configs for {" -> ".join(setup_dirs)}", **DEFAULT_PRED_STYLE
    )

    # get prediction iterations and setup names
    iterations = []
    setup_names = []
    for i, setup_dir in enumerate(setup_dirs):
        iteration = click.prompt(
            click.style(f"Enter checkpoint iteration for model {i+1}: {os.path.basename(setup_dir)}", **DEFAULT_PRED_STYLE),
            type=int,
            default=5000*len(volumes) if i == 0 else 3000,
            show_default=True,
        )
        iterations.append(iteration)
        setup_names.append(get_setup_name(setup_dir)) 

    num_gpus = click.prompt(
        click.style("Enter number of GPUs to use for prediction", **DEFAULT_PRED_STYLE),
        type=int,
        default=1,
    )
    num_workers = click.prompt(
        click.style("Enter number of CPU workers to use for prediction", **DEFAULT_PRED_STYLE),
        type=int,
        default=num_gpus,
    )

    # loop over volumes
    configs = {}
    for volume in volumes:
        pred_config = {}
        container = volume["output_container"]
        volume_name = volume["name"]
        raw_array = volume["raw_dataset"]

        click.echo()
        click.secho(
            f"Creating prediction configs for {volume_name}", **DEFAULT_PRED_STYLE
        )

        roi_offset, roi_shape, _ = get_sub_roi(in_array=raw_array)
        output_datasets = [] # list of lists of output datasets per setup per volume

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
                out_ds = [f"{out_ds_prefix}/{iteration}/{x}" for x in model_outputs]
            else:
                in_ds = [os.path.join(container, ds) for ds in output_datasets[-1]]
                out_ds = [f"{out_ds_prefix}/{iteration}--from--{chain_str}/{x}" for x in model_outputs]

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

        configs[volume_name] = check_and_update(pred_config, style=DEFAULT_PRED_STYLE)

    print(output_datasets)
    out_affs_ds = [ds for x in output_datasets for ds in x if ds.split('/')[-1].startswith("3d_affs")][-1]

    return {
        "out_affs_dataset": out_affs_ds, # final 3d affs dataset to segment
        "out_pred_datasets": [ds for x in output_datasets for ds in x], # sequence of pred datasets
        "configs": configs,
    }


def create_segmentation_configs(volumes, out_affs_ds):
    
    click.echo()
    click.secho(
        f"Creating Segmentation configs for {out_affs_ds}", **DEFAULT_SEG_STYLE
    )

    output_prefix = os.path.dirname(out_affs_ds)

    method, params = choose_seg_method_params()

    out_frags_ds = f"{output_prefix}/fragments_{method}"
    out_lut_dir = f"{output_prefix}/luts_{method}"
    out_seg_prefix = f"{output_prefix}/segmentations_{method}"

    configs = {}
    for volume in volumes:
        container = volume["output_container"]
        volume_name = volume["name"]
        affs_array = os.path.join(container, out_affs_ds)
        frags_array = os.path.join(container, out_frags_ds)
        lut_dir = os.path.join(container, out_lut_dir)

        click.echo()
        click.secho(
            f"Segmentation config for {volume_name}", **DEFAULT_SEG_STYLE
        )

        # TODO: find way to get roi from predictions
        # roi_offset, roi_shape, voxel_size = get_roi(in_array=affs_array)

        do_blockwise = False

        if click.confirm(
            click.style(f"Do blockwise = {do_blockwise}. Switch?", **DEFAULT_SEG_STYLE), default=False, show_default=True
        ):
            do_blockwise = not do_blockwise

        if do_blockwise and click.confirm(
            click.style(f"Set block shape and context?", **DEFAULT_SEG_STYLE), default=False, show_default=True
        ):
            block_shape = click.prompt(
                click.style("Enter block shape in voxels (e.g. 128,128,128), or 'roi' for single block with daisy", **DEFAULT_SEG_STYLE),
                #default="128,128,128",
                type=str,
            )
            context = click.prompt(
                click.style("Enter context in voxels (e.g. 128,128,128)", **DEFAULT_SEG_STYLE),
                #default="128,128,128",
                type=str,
            )
            if block_shape is not None and block_shape != "roi":
                block_shape = [int(x) for x in block_shape.split(",")]
            if context:
                context = [int(x) for x in context.split(",")]
            num_workers = click.prompt(click.style("Enter number of workers", **DEFAULT_SEG_STYLE), default=10, type=int)
        else:
            block_shape = None
            context = None
            num_workers = 1

        # are raw masks available ?
        if volume["raw_mask_dataset"] is not None:
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
            sqlite_path = os.path.join(
                container, f"{output_prefix}/rag_{method}.db"
            )

            # SQLite or not ?
            use_sqlite = not do_blockwise
            if click.confirm(
                click.style(f"Use SQLite for RAG = {use_sqlite}. Switch?", **DEFAULT_SEG_STYLE),
                default=False,
                show_default=True,
            ):
                use_sqlite = not use_sqlite

            sqlite_path = sqlite_path if use_sqlite else None

            # get rag db config
            seg_config["db"] = get_rag_db_config(sqlite_path) 

        configs[volume_name] = check_and_update(seg_config, style=DEFAULT_SEG_STYLE)

    return {"out_seg_prefix": out_seg_prefix, "configs": configs}


def create_evaluation_configs(volumes, out_seg_prefix, pred_datasets):
    
    click.echo()
    click.secho(
        f"Evaluation configs", **DEFAULT_EVAL_STYLE
    )

    output_prefix = os.path.dirname(out_seg_prefix)

    configs = {}
    for volume in volumes:
        volume_name = volume["name"]
        container = volume["output_container"]

        click.echo()
        click.secho(
            f"Creating evaluation config for {out_seg_prefix}", **DEFAULT_EVAL_STYLE
        )

        # gt labels evaluation ?
        if click.confirm(
            click.style(f"Are ground truth labels available for {volume_name}?", **DEFAULT_EVAL_STYLE),
            default=False,
            show_default=True,
        ):
            gt_labels_ds = os.path.abspath(
                click.prompt(
                    click.style(
                        "Enter path to ground truth labels dataset (press enter to skip) >>>",
                        **DEFAULT_EVAL_STYLE
                    ),
                    type=click.Path(exists=True, dir_okay=True, file_okay=False),
                    default=None,
                    show_default=True,
                )
            )
        else:
            gt_labels_ds = None

        # gt skeletons evaluation ?
        if click.confirm(
            click.style(f"Are ground truth skeletons available for {volume_name}?", **DEFAULT_EVAL_STYLE),
            default=False,
            show_default=True,
        ):
            gt_skeletons_file = os.path.abspath(
                click.prompt(
                    click.style("Enter path to ground truth skeletons file (.graphml format) (press enter to skip)", **DEFAULT_EVAL_STYLE),
                    type=click.Path(exists=True, dir_okay=False, file_okay=True),
                    default=None,
                    show_default=True,
                )
            )
        else:
            gt_skeletons_file = None

        # self pred evaluation ?
        if click.confirm(
            click.style(f"Compute prediction errors for {volume_name}?", **DEFAULT_EVAL_STYLE),
            default=True,
            show_default=True,
        ):
            pred_choices = [ds for ds in pred_datasets if ds.split('/')[-1].startswith("3d_")]

            if len(pred_choices) == 1:
                pred_ds = pred_choices[0]
            else:
                pred_ds = click.prompt(
                    click.style(f"Select {volume_name} predictions to self-evaluate with:", **DEFAULT_EVAL_STYLE),
                    type=click.Choice(pred_choices),
                    default=pred_choices[-1],
                    show_default=True,
                    show_choices=True,
                )

            pred_type = pred_ds.split('/')[-1][3:7]
            assert pred_type in ["lsds", "affs"]

        else:
            pred_ds = None

        # are raw masks available ?
        if volume["raw_mask_dataset"] is not None:
            mask_dataset = volumes["raw_mask_dataset"]
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
            eval_config["self"] = {
                "pred_dataset": os.path.join(container, pred_ds),
                "thresholds": [0.1, 1.0],
            }

        if gt_labels_ds is not None or gt_skeletons_file is not None:
            eval_config["gt"] = {
                "labels_dataset": gt_labels_ds,
                "skeletons_file": gt_skeletons_file,
            }

        configs[volume_name] = check_and_update(eval_config, style=DEFAULT_EVAL_STYLE)

    return {"out_eval_dir": output_prefix, "configs": configs}


def create_filter_configs(volumes, in_seg_prefix, eval_dir):

    click.echo()
    click.secho(
        f"Filter configs", **DEFAULT_FILTER_STYLE
    )

    out_seg_ds_prefix = in_seg_prefix.replace("/segmentations_", "/pseudo_gt_ids_")
    out_mask_ds_prefix = in_seg_prefix.replace("/segmentations_", "/pseudo_gt_mask_")

    out_volumes = []

    configs = {}
    for volume in volumes:
        container = volume["output_container"]
        volume_name = volume["name"]
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

        configs[volume_name] = check_and_update(filter_config, style=DEFAULT_FILTER_STYLE)

        # volumes for the next round
        out_volumes.append(
            {
                "name": volume_name,
                "raw_dataset": volume["raw_dataset"],
                "raw_mask_dataset": volume["raw_mask_dataset"],
                "labels_dataset": out_seg_ds,
                "labels_mask_dataset": out_mask_ds,
                "voxel_size": volume["voxel_size"],
                "previous_labels_datasets": [volume['labels_dataset'],] + volume.get("previous_labels_datasets", []),
                "previous_labels_mask_datasets": [volume['labels_mask_dataset'],] + volume.get("previous_labels_mask_datasets", [])
            }
        )

    return {"out_volumes": out_volumes, "configs": configs}


def make_round_configs(volumes, round_dir):
    """Create all configs for a model with given volumes."""

    run_dir = click.prompt(
        click.style("Enter run directory", **DEFAULT_PROMPT_STYLE),
        default=os.path.join(round_dir, "run"),
        show_default=True,
    )
    os.makedirs(run_dir, exist_ok=True)

    # training configs
    train_config = create_training_config(volumes, round_dir)
    for i, setup_dir in enumerate(train_config["configs"]):
        save_config(
            train_config["configs"][setup_dir],
            os.path.join(run_dir, f"01_train_{str(i).zfill(2)}.yaml"),
        )

    setup_dirs = train_config["setup_dirs"]
    pred_config = create_prediction_configs(volumes, setup_dirs)
    for volume_name in pred_config["configs"]:
        save_config(
            pred_config["configs"][volume_name],
            os.path.join(run_dir, f"02_pred_{volume_name}.yaml"),
        )

    out_affs_ds = pred_config["out_affs_dataset"]
    out_pred_datasets = pred_config["out_pred_datasets"]
    seg_configs = create_segmentation_configs(volumes, out_affs_ds)
    for volume_name in pred_config["configs"]:
        save_config(
            seg_configs["configs"][volume_name],
            os.path.join(run_dir, f"03_seg_{volume_name}.yaml"),
        )

    out_seg_prefix = seg_configs["out_seg_prefix"]
    eval_configs = create_evaluation_configs(
        volumes, out_seg_prefix, out_pred_datasets
    )
    for volume_name in pred_config["configs"]:
        save_config(
            eval_configs["configs"][volume_name],
            os.path.join(run_dir, f"04_eval_{volume_name}.yaml"),
        )

    out_eval_dir = eval_configs["out_eval_dir"]
    filter_configs = create_filter_configs(volumes, out_seg_prefix, out_eval_dir)
    for volume_name in pred_config["configs"]:
        save_config(
            filter_configs["configs"][volume_name],
            os.path.join(run_dir, f"05_filter_{volume_name}.yaml"),
        )

    out_volumes = filter_configs["out_volumes"]
    return out_volumes
