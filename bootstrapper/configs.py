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



DEFAULT_PROMPT_STYLE = {"fg": "cyan"}
DEFAULT_INFO_STYLE = {"fg": "cyan", "bold": True}
DEFAULT_TRAIN_STYLE = {"fg": "green"}
DEFAULT_PRED_STYLE = {"fg": "yellow"}
DEFAULT_SEG_STYLE = {"fg": "red"}
DEFAULT_EVAL_STYLE = {"fg": "magenta"}
DEFAULT_FILTER_STYLE = {"fg": "blue"}

BS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
MODEL_DIR = os.path.join(BS_DIR, "models")
MODEL_URLS = {
    "3d_affs_from_2d_affs": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.1.0/3d_affs_from_2d_affs.zip",
    "3d_affs_from_2d_lsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.1.0/3d_affs_from_2d_lsd.zip",
    "3d_affs_from_2d_mtlsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.1.0/3d_affs_from_2d_mtlsd.zip",
    "3d_affs_from_3d_lsd": "https://github.com/ucsdmanorlab/bootstrapper/releases/download/v0.1.0/3d_affs_from_3d_lsd.zip",
}

def check_and_update(configs, style=DEFAULT_PROMPT_STYLE):
    is_single = not isinstance(configs, list)
    configs = [configs] if is_single else configs

    click.echo()
    for config in configs:
        click.secho(pprint(config))

    if not click.confirm(
        click.style("Enter confirmation for values above (y/n)", **style), default=True
    ):
        if edited_yaml := click.edit(yaml.dump_all(configs)):
            configs = list(yaml.safe_load_all(edited_yaml))

    return configs[0] if is_single else configs


def save_config(config, filename):
    with open(filename, "w") as f:
        yaml.dump(config, f)
    click.secho(f"{filename} saved successfully.", **DEFAULT_INFO_STYLE)


def copy_model_scripts(model_name, setup_dir):
    src = os.path.abspath(os.path.join(BS_DIR, "models", model_name))
    click.secho(f"Copying {src} to {setup_dir}", **DEFAULT_TRAIN_STYLE)
    copytree(src, setup_dir, dirs_exist_ok=True)


def get_roi(in_array, offset=None, shape=None):
    """Get desired ROI within volume."""
    in_array = open_ds(in_array)
    full_roi = in_array.roi
    voxel_size = in_array.voxel_size
    full_shape = [s // v for s, v in zip(in_array.roi.shape, voxel_size)]

    if offset is None:
        offset = click.prompt(
            click.style(
                f"Enter voxel offset as space-separated integers in {in_array.axis_names}",
                **DEFAULT_PRED_STYLE,
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
                **DEFAULT_PRED_STYLE,
            ),
            default="0 0 0",
        )
        shape = tuple(map(int, shape.strip().split())) if shape != "0 0 0" else None

    roi_offset = [x * y for x, y in zip(offset, voxel_size)]

    if shape is None and roi_offset == [0, 0, 0]:
        roi_shape = [x * y for x, y in zip(full_shape, voxel_size)]
    else:
        remaining_shape = [
            fs - (ro // vs) for fs, ro, vs in zip(full_shape, roi_offset, voxel_size)
        ]
        if shape is None:
            roi_shape = [rem * vs for rem, vs in zip(remaining_shape, voxel_size)]
        else:
            roi_shape = [x * y for x, y in zip(shape, voxel_size)]
            roi_shape = [
                min(rs, rem * vs)
                for rs, rem, vs in zip(roi_shape, remaining_shape, voxel_size)
            ]

    roi = Roi(roi_offset, roi_shape)
    if not full_roi.contains(roi):
        click.secho(
            "ROI is not contained within the full volume's ROI. Cropping to..",
            **DEFAULT_PRED_STYLE,
        )
        roi = roi.intersect(full_roi)
        click.secho(f"{roi}", **DEFAULT_PRED_STYLE)

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
        click.style(f"Enter first model name", **DEFAULT_TRAIN_STYLE),
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

    # move setup_dir/model_name/checkpoint_*.pth to setup_dir/checkpoint_*.pth
    for file in os.listdir(os.path.join(setup_dir, model_name)):
        if "model_checkpoint_" in file:
            os.rename(
                os.path.join(setup_dir, model_name, file),
                os.path.join(setup_dir, file)
            )

    # clean up
    os.rmdir(os.path.join(setup_dir, model_name))
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
            train_config['samples'] = {
                v["zarr_container"]: {
                    "raw": v["raw_dataset"],
                    "labels": v["labels_dataset"],
                    "mask": v["labels_mask_dataset"],
                }
                for v in volumes
                if v["labels_dataset"] is not None
            }

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

    # get prediction iterations
    iterations = []
    for i, setup_dir in enumerate(setup_dirs):
        iteration = click.prompt(
            click.style(f"Enter checkpoint iteration for model {i+1}: {os.path.basename(setup_dir)}", **DEFAULT_PRED_STYLE),
            type=int,
            default=5000*len(volumes) if i == 0 else 3000,
            show_default=True,
        )
        iterations.append(iteration)

    num_gpus = click.prompt(
        click.style("Enter number of GPUs to use for prediction", **DEFAULT_PRED_STYLE),
        type=int,
        default=1,
    )

    num_workers = click.prompt(
        click.style("Enter number of CPU workers to use for prediction", **DEFAULT_PRED_STYLE),
        type=int,
        default=1,
    )

    # loop over volumes
    configs = {}
    for volume in volumes:
        pred_config = {}
        container = volume["zarr_container"]
        volume_name = os.path.basename(container).split(".zarr")[0]
        raw_array = volume["raw_dataset"]

        click.echo()
        click.secho(
            f"Creating prediction configs for {volume_name}", **DEFAULT_PRED_STYLE
        )

        roi_offset, roi_shape, _ = get_roi(in_array=raw_array)
        output_datasets = [] # list of lists of output datasets per setup

        for i, setup_dir in enumerate(setup_dirs):
            iteration = iterations[i]
            setup_name = os.path.basename(setup_dir)

            # get model outputs
            with open(os.path.join(setup_dir, "net_config.json"), "r") as f:
                model_outputs = json.load(f)["outputs"]

            # get in and out dataset names
            out_ds_prefix = f"predictions/{setup_name}"
            if i == 0:
                in_ds = [raw_array]
                out_ds = [f"{out_ds_prefix}/{x}_{iteration}" for x in model_outputs]
            else:
                chain = []
                for j in range(i-1, -1, -1):
                    prev_setup = os.path.basename(setup_dirs[j])
                    prev_iteration = iterations[j]
                    chain.append(f"{prev_setup}_{prev_iteration}")

                chain_str = "-from-".join(chain)
                in_ds = [os.path.join(container, ds) for ds in output_datasets[-1]]
                out_ds = [f"{out_ds_prefix}/{x}_{iteration}-from-{chain_str}" for x in model_outputs]

            output_datasets.append(out_ds)

            pred_config[f"{str(i+1).zfill(2)}_{setup_name}"] = {
                "setup_dir": setup_dir,
                "input_datasets": in_ds,
                "roi_offset": list(roi_offset),
                "roi_shape": list(roi_shape),
                "checkpoint": os.path.join(setup_dir, f"model_checkpoint_{iteration}"),
                "output_container": container,
                "output_datasets_prefix": out_ds_prefix,
                "num_workers": num_workers,
                "num_gpus": num_gpus,
            }

        configs[volume_name] = check_and_update(pred_config, style=DEFAULT_PRED_STYLE)

    print(output_datasets)
    out_affs_ds = [ds for x in output_datasets for ds in x if ds.split('/')[-1].startswith("3d_affs")][-1]

    return {
        "out_affs_dataset": out_affs_ds, # final 3d affs dataset to segment
        "out_pred_datasets": output_datasets, # sequence of pred datasets
        "configs": configs,
    }


def create_segmentation_configs(volumes, out_affs_ds, setup_dir=None):
    
    click.echo()
    click.secho(
        f"Creating Segmentation configs for {out_affs_ds}", **DEFAULT_SEG_STYLE
    )

    if setup_dir is not None:
        setup_name = os.path.basename(setup_dir)
    else:
        setup_name = click.prompt(
            click.style("Enter setup name for segmentations", **DEFAULT_SEG_STYLE),
            default=out_affs_ds.split("/")[-2],
            show_default=True,
        )

    # TODO: add support for choice between waterz, mws, thresh
    waterz_defaults = {
        "fragments_in_xy": True,
        "min_seed_distance": 10,
        "epsilon_agglomerate": 0.0,
        "filter_fragments": 0.05,
        "replace_sections": None,
        "thresholds_minmax": [0, 1],
        "thresholds_step": 0.05,
        "thresholds": [0.2, 0.3],
        "merge_function": "mean",
    }

    out_frags_ds = f"post/{setup_name}/fragments"
    out_lut_dir = f"post/{setup_name}/luts"
    out_seg_prefix = f"post/{setup_name}/segmentations"

    configs = {}
    for volume in volumes:

        volume_name = os.path.basename(volume["zarr_container"]).split(".zarr")[0]
        affs_array = os.path.join(volume["zarr_container"], out_affs_ds)

        click.echo()
        click.secho(
            f"Segmentation config for {volume_name}", **DEFAULT_SEG_STYLE
        )

        # TODO: find way to get roi from predictions
        # roi_offset, roi_shape, voxel_size = get_roi(in_array=affs_array)

        do_blockwise = True

        if click.confirm(
            click.style(f"Do blockwise = {do_blockwise}. Switch?", **DEFAULT_SEG_STYLE), default=False, show_default=True
        ):
            do_blockwise = not do_blockwise

        if do_blockwise and click.confirm(
            click.style(f"Set block shape and context?", **DEFAULT_SEG_STYLE), default=False, show_default=True
        ):
            block_shape = click.prompt(
                click.style("Enter block shape in voxels (e.g. 128,128,128)", **DEFAULT_SEG_STYLE),
                default="128,128,128",
                type=str,
            )
            context = click.prompt(
                click.style("Enter context in voxels (e.g. 128,128,128)", **DEFAULT_SEG_STYLE),
                default="128,128,128",
                type=str,
            )
            block_shape = [int(x) for x in block_shape.split(",")]
            context = [int(x) for x in context.split(",")]
            num_workers = click.prompt(click.style("Enter number of workers", **DEFAULT_SEG_STYLE), default=10, type=int)
        else:
            block_shape = None
            context = None
            num_workers = 1

        sqlite_path = os.path.join(
            volume["zarr_container"], f"post/{setup_name}/rag.db"
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
        db_config = get_rag_db_config(sqlite_path)

        # are raw masks available ?
        if volume["raw_mask_dataset"] is not None:
            mask_file = volume["zarr_container"]
            mask_dataset = volumes["raw_mask_dataset"]
        else:
            mask_file = None
            mask_dataset = None

        waterz_config = {
            "affs_file": volume["zarr_container"],
            "affs_dataset": out_affs_ds,
            "fragments_file": volume["zarr_container"],
            "fragments_dataset": out_frags_ds,
            "lut_dir": out_lut_dir,
            "seg_file": volume["zarr_container"],
            "seg_dataset_prefix": out_seg_prefix,
            "mask_file": mask_file,
            "mask_dataset": mask_dataset,
            # "roi_offset": roi_offset,
            # "roi_shape": roi_shape,
            "block_shape": block_shape,
            "context": block_shape,
            "blockwise": do_blockwise,
            "num_workers": num_workers,
        } | waterz_defaults

        seg_config = {"db": db_config, "waterz": waterz_config}
        configs[volume_name] = check_and_update(seg_config, style=DEFAULT_SEG_STYLE)

    return {"out_seg_prefix": out_seg_prefix, "configs": configs}


def create_evaluation_configs(volumes, out_seg_prefix, pred_datasets, setup_dir=None):
    
    click.echo()
    click.secho(
        f"Evaluation configs", **DEFAULT_EVAL_STYLE
    )

    if setup_dir is not None:
        setup_name = os.path.basename(setup_dir)
    else:
        setup_name = click.prompt(
            click.style("Enter setup name for evaluation outputs", **DEFAULT_SEG_STYLE),
            default=out_seg_prefix.split("/")[-2],
            show_default=True,
        )

    out_eval_dir = f"post/{setup_name}/eval"

    configs = {}
    for volume in volumes:
        volume_name = os.path.basename(volume["zarr_container"]).split(".zarr")[0]
        container = volume["zarr_container"]

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
            pred_choices = [x for x in pred_datasets if x.split('/')[2].startswith("3d_")]

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

            pred_type = pred_ds.split('/')[2][3:7]
            assert pred_type in ["lsds", "affs"]

            pred_error_map_ds = os.path.join(
                container, out_eval_dir, f"{pred_type}_error_map"
            )
            pred_error_mask_ds = os.path.join(
                container, out_eval_dir, f"{pred_type}_error_mask"
            )

        else:
            pred_ds = None
            pred_error_map_ds = None
            pred_error_mask_ds = None

        # add evaluation mask ? #TODO
        mask_file = None
        mask_dataset = None

        # get evaluation ROI TODO: get from segmentation config
        # roi_offset, roi_shape, voxel_size = get_roi(in_array=out_segs)

        eval_config = {
            "out_dir": out_eval_dir,
            "seg_file": container,
            "seg_datasets": out_seg_prefix,  # TODO: zarr tree find all seg arrays. eval on all.
            "mask_file": mask_file,
            "mask_dataset": mask_dataset,
            "fragments_file": container,
            # "roi_offset": roi_offset,
            # "roi_shape": roi_shape,
        }

        if pred_ds is not None:
            eval_config["self"] = {
                "pred_dataset": pred_ds,
                "out_map_dataset": pred_error_map_ds,
                "out_mask_dataset": pred_error_mask_ds,
                "thresholds": [0.1, 1.0],
            }

        if gt_labels_ds is not None or gt_skeletons_file is not None:
            eval_config["gt"] = {
                "gt_labels_dataset": gt_labels_ds,
                "gt_skeletons_file": gt_skeletons_file,
            }

        configs[volume_name] = check_and_update(eval_config, style=DEFAULT_EVAL_STYLE)

    return {"out_eval_dir": out_eval_dir, "configs": configs}


def create_filter_configs(volumes, out_seg_prefix, eval_dir, setup_dir=None):

    click.echo()
    click.secho(
        f"Filter configs", **DEFAULT_FILTER_STYLE
    )

    if setup_dir is not None:
        setup_name = os.path.basename(setup_dir)
    else:
        setup_name = click.prompt(
            click.style("Enter setup name for filtered segmentations", **DEFAULT_SEG_STYLE),
            default=out_seg_prefix.split("/")[-2],
            show_default=True,
        )

    out_seg_ds = f"pseudo_gt/{setup_name}/ids"
    out_mask_ds = f"pseudo_gt/{setup_name}/mask"

    out_volumes = []

    configs = {}
    for volume in volumes:
        volume_name = os.path.basename(volume["zarr_container"]).split(".zarr")[0]

        # get filter ROI TODO: get from eval config
        # roi_offset, roi_shape, _ = get_roi(in_array=out_segs)

        filter_config = {
            "seg_file": volume["zarr_container"],
            "seg_datasets": out_seg_prefix,
            "eval_dir": eval_dir,
            "out_file": volume["zarr_container"],
            "out_seg_dataset": out_seg_ds,
            "out_mask_dataset": out_mask_ds,
            # "roi_offset": roi_offset,
            # "roi_shape": roi_shape,
            "dust_filter": 500,
            "remove_outliers": True,
            "remove_z_fragments": 10,
            "overlap_filter": 0.0,
            "erode_out_mask": False,
        }

        configs[volume_name] = check_and_update(filter_config, style=DEFAULT_FILTER_STYLE)

        out_volumes.append(
            {
                "zarr_container": volume["zarr_container"],
                "raw_dataset": volume["raw_dataset"],
                "raw_mask_dataset": volume["raw_mask_dataset"],
                "labels_dataset": out_seg_ds,
                "labels_mask_dataset": out_mask_ds,
                "voxel_size": volume["voxel_size"],
                "previous_volume": volume,
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
    for setup_dir in train_config["configs"]:
        save_config(
            train_config["configs"][setup_dir],
            os.path.join(run_dir, f"train_{os.path.basename(setup_dir)}.yaml"),
        )

    setup_dirs = train_config["setup_dirs"]
    pred_config = create_prediction_configs(volumes, setup_dirs)
    for volume_name in pred_config["configs"]:
        save_config(
            pred_config["configs"][volume_name],
            os.path.join(run_dir, f"pred_{volume_name}.yaml"),
        )

    out_affs_ds = pred_config["out_affs_dataset"]
    out_pred_datasets = pred_config["out_pred_datasets"]
    seg_configs = create_segmentation_configs(volumes, out_affs_ds)
    for volume_name in pred_config["configs"]:
        save_config(
            seg_configs["configs"][volume_name],
            os.path.join(run_dir, f"seg_{volume_name}.yaml"),
        )

    out_seg_prefix = seg_configs["out_seg_prefix"]
    eval_configs = create_evaluation_configs(
        volumes, out_seg_prefix, out_pred_datasets
    )
    for volume_name in pred_config["configs"]:
        save_config(
            eval_configs["configs"][volume_name],
            os.path.join(run_dir, f"eval_{volume_name}.yaml"),
        )

    out_eval_dir = eval_configs["out_eval_dir"]
    filter_configs = create_filter_configs(volumes, out_seg_prefix, out_eval_dir)
    for volume_name in pred_config["configs"]:
        save_config(
            filter_configs["configs"][volume_name],
            os.path.join(run_dir, f"filter_{volume_name}.yaml"),
        )

    out_volumes = filter_configs["out_volumes"]
    return out_volumes
