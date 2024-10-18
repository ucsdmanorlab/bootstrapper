import click
import yaml
import json
import os
import logging
from shutil import copytree
from pprint import pprint

from funlib.geometry import Roi
from funlib.persistence import open_ds

this_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

        
def check_and_update(configs):
    is_single = not isinstance(configs, list)
    configs = [configs] if is_single else configs

    for config in configs:
        logger.info(pprint(config))

    if not click.confirm("Are these values good?", default=True):
        if edited_yaml := click.edit(yaml.dump_all(configs)):
            configs = list(yaml.safe_load_all(edited_yaml))

    return configs[0] if is_single else configs


def save_config(config, filename):
    with open(filename, "w") as f:
        yaml.dump(config, f)
    logger.info(f"{filename} saved successfully.")


def copy_model_scripts(model_name, setup_dir):
    src = os.path.abspath(os.path.join(this_dir, "models", model_name))
    logger.info(f"Copying {src} to {setup_dir}")
    copytree(src, setup_dir, dirs_exist_ok=True)


def get_roi(in_array, offset=None, shape=None):
    """Get desired ROI within volume."""
    in_array = open_ds(in_array)
    full_roi = in_array.roi
    voxel_size = in_array.voxel_size
    full_shape = [s // v for s, v in zip(in_array.roi.shape, voxel_size)]

    if offset is None:
        offset = click.prompt(
            f"Enter voxel offset as space-separated integers in {in_array.axis_names}",
            type=str,
            default="0 0 0",
            show_default=False,
        )
        offset = tuple(map(int, offset.strip().split()))

    if shape is None:
        shape = click.prompt(
            f"Enter required voxel shape starting from {offset} as space-separated integers in {in_array.axis_names} (skipping will get remaining available shape)",
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
        logger.info("ROI is not contained within the full volume's ROI. Cropping to..")
        roi = roi.intersect(full_roi)
        logger.info(f"{roi}")

    return roi.offset, roi.shape, voxel_size


def get_rag_db_config(sqlite_path=None):
    nodes_table = click.prompt(
        "Enter RAG nodes table name", default="nodes", show_default=True
    )
    edges_table = click.prompt(
        "Enter RAG edges table name", default="edges", show_default=True
    )

    if sqlite_path:
        db_file = click.prompt(
            f"Enter SQLite RAG database file", default=sqlite_path, show_default=True
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
            logger.info(
                "PgSQL Database credentials not found in environment variables."
            )
            db_host = click.prompt("Enter PgSQL RAG database host")
            db_user = click.prompt("Enter PgSQL RAG database user")
            db_password = click.prompt(
                "Enter PgSQL RAG database password (input is hidden)", hide_input=True
            )
            db_port = click.prompt("Enter PgSQL RAG database port", type=int)

        db_name = click.prompt("Enter PgSQL RAG database name")
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


def choose_model():
    models = sorted(
        [
            d
            for d in os.listdir(os.path.join(os.path.dirname(__file__), "models"))
            if os.path.isdir(os.path.join(os.path.dirname(__file__), "models", d))
            and "_from_" not in d
        ]
    )
    model = click.prompt(
        f"Enter model name:",
        type=click.Choice(models),
        show_choices=True,
    )

    return model


def create_training_config(volumes, setup_dir, model_name):
    logger.info(f"\nTraining config for {model_name} in {setup_dir}:")
    setup_dir = os.path.abspath(setup_dir)
    copy_model_scripts(model_name, setup_dir)
    os.makedirs(os.path.join(setup_dir, "run"), exist_ok=True)

    # get voxel size from volumes, assume all volumes have same voxel size
    voxel_size = volumes[0]["voxel_size"]

    max_iterations = click.prompt("Enter max iterations", default=30001, type=int)
    save_checkpoints_every = click.prompt(
        "Enter save checkpoints every", default=5000, type=int
    )
    save_snapshots_every = click.prompt(
        "Enter save snapshots every", default=1000, type=int
    )

    training_samples = {
        v["zarr_container"]: {
            "raw": v["raw_dataset"],
            "labels": v["labels_dataset"],
            "mask": v["labels_mask_dataset"],
        }
        for v in volumes
        if v["labels_dataset"] is not None
    }

    train_config = {
        "setup_dir": setup_dir,
        "samples": training_samples,
        "voxel_size": voxel_size,
        "out_dir": setup_dir,
        "max_iterations": max_iterations,
        "save_checkpoints_every": save_checkpoints_every,
        "save_snapshots_every": save_snapshots_every,
    }

    if "lsd" in model_name.lower():
        train_config["sigma"] = click.prompt(
            "Enter sigma for LSD model", default=10 * voxel_size[-1], type=int
        )

    train_config = check_and_update(train_config)

    return train_config


def create_prediction_configs(volumes, setup_dir):

    model_name = os.path.basename(setup_dir)
    round_name = os.path.basename(os.path.dirname(setup_dir))

    logger.info(f"\nPrediction configs for {round_name}/{model_name}:")

    pred_iter = click.prompt(
        "Enter checkpoint iteration for prediction",
        type=int,
        default=5000,
    )

    # get model outputs
    with open(os.path.join(setup_dir, "net_config.json"), "r") as f:
        model_outputs = json.load(f)["outputs"]

    pred_datasets = [
        f"predictions/{round_name}-{model_name}/{x}_{pred_iter}" for x in model_outputs
    ]

    # make 3d_affs using model trained on synthetic data ?
    make_affs = "3d_affs" not in model_outputs
    if make_affs:
        affs_setup_dir = os.path.abspath(
            os.path.join(
                this_dir, f"models/3d_affs_from_{model_name.replace('_3ch','')}"
            )
        )
        affs_iter = click.prompt(
            f"Enter checkpoint iteration for inference of affs with {affs_setup_dir}",
            default=5000,
            type=int,
        )
        out_affs_ds = f"predictions/{round_name}-{model_name}/3d_affs_{affs_iter}_from_{pred_datasets[0].split('_')[-1]}"
    else:
        out_affs_ds = [x for x in pred_datasets if "3d_affs" in x][0]

    # can lsd errors be computed ?
    # computable_pred_errors = ["3d_affs"]True if "3d_lsds" in model_outputs else False

    configs = {}
    for volume in volumes:
        pred_config = {}
        container = volume["zarr_container"]
        volume_name = os.path.basename(container).split(".zarr")[0]
        raw_array = volume["raw_dataset"]

        logger.info(f"\nCreating prediction config for {raw_array}:")

        roi_offset, roi_shape, _ = get_roi(in_array=raw_array)

        pred_config[model_name] = {
            "setup_dir": setup_dir,
            "input_datasets": [raw_array],
            "roi_offset": list(roi_offset),
            "roi_shape": list(roi_shape),
            "checkpoint": os.path.join(setup_dir, f"model_checkpoint_{pred_iter}"),
            "output_container": container,
            "output_datasets_prefix": f"predictions/{round_name}-{model_name}",
            "num_workers": 4,
            "num_gpus": 4,
            "num_cache_workers": 1,
        }

        if make_affs:
            pred_config["affs"] = {
                "setup_dir": affs_setup_dir,
                "input_datasets": [os.path.join(container, ds) for ds in pred_datasets],
                "roi_offset": list(roi_offset),
                "roi_shape": list(roi_shape),
                "checkpoint": os.path.join(
                    affs_setup_dir, f"model_checkpoint_{affs_iter}"
                ),
                "output_container": container,
                "output_datasets_prefix": f"predictions/{round_name}-{model_name}",
                "num_workers": 4,
                "num_gpus": 4,
                "num_cache_workers": 1,
            }

        configs[volume_name] = check_and_update(pred_config)

    return {
        "out_affs_dataset": out_affs_ds,
        "out_pred_datasets": pred_datasets,
        "configs": configs,
    }


def create_segmentation_configs(volumes, setup_dir, out_affs_ds):

    round_name = os.path.basename(os.path.dirname(setup_dir))
    model_name = os.path.basename(setup_dir)
    logger.info(f"\nSegmentation configs for {round_name}/{model_name}:")

    waterz_defaults = {
        "fragments_in_xy": True,
        "min_seed_distance": 10,
        "epsilon_agglomerate": 0.0,
        "filter_fragments": 0.05,
        "replace_sections": None,
        "thresholds_minmax": [0, 1],
        "thresholds_step": 0.05,
        "thresholds": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65],
        "merge_function": "mean",
    }

    configs = {}
    for volume in volumes:

        volume_name = os.path.basename(volume["zarr_container"]).split(".zarr")[0]
        affs_array = os.path.join(volume["zarr_container"], out_affs_ds)
        out_frags_ds = f"post/{round_name}-{model_name}/fragments"
        out_lut_dir = f"post/{round_name}-{model_name}/luts"
        out_seg_ds = f"post/{round_name}-{model_name}/segmentations"

        logger.info(f"\nCreating segmentation config for {affs_array}:")

        # TODO: find way to get roi from predictions
        # roi_offset, roi_shape, voxel_size = get_roi(in_array=affs_array)

        do_blockwise = True

        if click.confirm(
            f"Do blockwise = {do_blockwise}. Switch?", default=False, show_default=True
        ):
            do_blockwise = not do_blockwise

        if do_blockwise and click.confirm(
            f"Set block shape and context?", default=False, show_default=True
        ):
            block_shape = click.prompt(
                "Enter block shape in voxels (e.g. 128,128,128)",
                default="128,128,128",
                type=str,
            )
            context = click.prompt(
                "Enter context in voxels (e.g. 128,128,128)",
                default="128,128,128",
                type=str,
            )
            block_shape = [int(x) for x in block_shape.split(",")]
            context = [int(x) for x in context.split(",")]
            num_workers = click.prompt("Enter number of workers", default=10, type=int)
        else:
            block_shape = None
            context = None
            num_workers = 1

        sqlite_path = os.path.join(
            volume["zarr_container"], f"post/{round_name}-{model_name}/rag.db"
        )

        # SQLite or not ?
        use_sqlite = not do_blockwise
        if click.confirm(
            f"Use SQLite for RAG = {use_sqlite}. Switch?",
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
            "seg_dataset": out_seg_ds,
            "mask_file": mask_file,
            "mask_dataset": mask_dataset,
            # "roi_offset": roi_offset,
            # "roi_shape": roi_shape,
            "block_shape": block_shape,
            "context": block_shape,
            "blockwise": do_blockwise,
            "num_workers": num_workers,
        } | waterz_defaults

        seg_config = {
            "db": db_config,
            "waterz": waterz_config
        }
        configs[volume_name] = check_and_update(seg_config)


    return {
        "out_seg_dataset": out_seg_ds,
        "configs": configs
    }


def create_evaluation_configs(volumes, setup_dir, out_segs, pred_datasets):

    round_name = os.path.basename(os.path.dirname(setup_dir))
    model_name = os.path.basename(setup_dir)
    logger.info(f"\nEvaluation configs for {round_name}/{model_name}:")

    out_eval_dir = f"post/{round_name}-{model_name}/eval"

    configs = {}
    for volume in volumes:
        volume_name = os.path.basename(volume["zarr_container"]).split(".zarr")[0]
        container = volume["zarr_container"]

        logger.info(f"\nCreating evaluation config for {out_segs}:")

        # gt evaluation ?
        if click.confirm(f"Are ground truth labels available for {out_segs}?", default=False, show_default=True):
            gt_labels_ds = os.path.abspath(click.prompt(
                "Enter path to ground truth labels dataset (press enter to skip)",
                type=click.Path(exists=True, dir_okay=True, file_okay=False),
                default=None,
                show_default=True
            ))
        else:
            gt_labels_ds = None

        if click.confirm(f"Are ground truth skeletons available for {out_segs}?", default=False, show_default=True):
            gt_skeletons_file = os.path.abspath(click.prompt(
                "Enter path to ground truth skeletons file (.graphml format) (press enter to skip)",
                type=click.Path(exists=True, dir_okay=False, file_okay=True),
                default=None,
                show_default=True
            ))
        else:
            gt_skeletons_file = None

        # self evaluation ?
        if click.confirm(
            f"Compute prediction errors for {out_segs}?", default=True, show_default=True
        ):
            # on what predictions? lsds, affs, or both?
            pred_choices = ["lsds", "affs"] if len([x for x in pred_datasets if "3d_lsds" in x and "_from_" not in x]) > 0 else ["affs"]
            pred_choice = click.prompt(
                "Select predictions to self-evaluate with:",
                type=click.Choice(pred_choices), #TODO: support "both"
                default=pred_choices[0],
                show_default=True,
                show_choices=True,
            )

            if pred_choice == "lsds":
                pred_ds = [
                    os.path.join(container,x) for x in pred_datasets if "3d_lsds" in x and "_from_" not in x
                ][0]
            else:
                pred_ds = [
                    os.path.join(container,x) for x in pred_datasets if "3d_affs" in x
                ]
                if len(pred_ds) > 1:
                    logger.info(f"Multiple 3d_affs datasets found: {pred_ds}")
                    pred_ds = click.Choice(pred_ds)
                else:
                    pred_ds = pred_ds[0]
        
            pred_error_map_ds = os.path.join(container, out_eval_dir, f"{pred_choice}_error_map")
            pred_error_mask_ds = os.path.join(container, out_eval_dir, f"{pred_choice}_error_mask")

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
            "seg_datasets": out_segs,  # TODO: zarr tree find all seg arrays. eval on all.
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

        configs[volume_name] = check_and_update(eval_config)

    return {
        "out_eval_dir": out_eval_dir,
        "configs": configs
    }


def create_filter_configs(volumes, setup_dir, out_segs, eval_dir):

    model_name = os.path.basename(setup_dir)
    round_name = os.path.basename(os.path.dirname(setup_dir))

    logger.info(f"\nFilter configs for {round_name}/{model_name}:")

    out_seg_ds = f"pseudo_gt/{round_name}-{model_name}/ids"
    out_mask_ds = f"pseudo_gt/{round_name}-{model_name}/mask"

    out_volumes = []

    configs = {}
    for volume in volumes:
        volume_name = os.path.basename(volume["zarr_container"]).split(".zarr")[0]

        # get filter ROI TODO: get from eval config
        # roi_offset, roi_shape, _ = get_roi(in_array=out_segs)

        filter_config = {
            "seg_file": volume["zarr_container"],
            "seg_datasets": out_segs,
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

        configs[volume_name] = check_and_update(filter_config)

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

    return {
        "out_volumes": out_volumes,
        "configs": configs
    }


def make_round_configs(volumes, setup_dir, model_name):
    """Create all configs for a model with given volumes."""

    train_config = create_training_config(volumes, setup_dir, model_name)

    setup_dir = train_config['setup_dir']
    pred_config = create_prediction_configs(volumes, setup_dir)

    out_affs_ds = pred_config['out_affs_dataset']
    out_pred_datasets = pred_config['out_pred_datasets']
    seg_configs = create_segmentation_configs(volumes, setup_dir, out_affs_ds)

    out_seg_ds = seg_configs['out_seg_dataset']
    eval_configs = create_evaluation_configs(
        volumes, setup_dir, out_seg_ds, out_pred_datasets + [out_affs_ds]
    )
    
    out_eval_dir = eval_configs['out_eval_dir']
    filter_configs = create_filter_configs(
        volumes,
        setup_dir,
        out_seg_ds,
        out_eval_dir
    )

    # write configs
    run_dir = os.path.join(setup_dir, "run")
    os.makedirs(run_dir, exist_ok=True)
    save_config(train_config, os.path.join(run_dir, "train.yaml"))
    for volume_name in pred_config['configs']:
        save_config(
            pred_config['configs'][volume_name],
            os.path.join(run_dir, f"pred_{volume_name}.yaml"),
        )
        save_config(
            seg_configs['configs'][volume_name],
            os.path.join(run_dir, f"seg_{volume_name}.yaml"),
        )
        save_config(
            eval_configs['configs'][volume_name],
            os.path.join(run_dir, f"eval_{volume_name}.yaml"),
        )
        save_config(
            filter_configs['configs'][volume_name],
            os.path.join(run_dir, f"filter_{volume_name}.yaml"),
        )

    out_volumes = filter_configs['out_volumes']
    return out_volumes
