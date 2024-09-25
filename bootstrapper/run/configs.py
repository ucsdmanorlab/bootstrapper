import click
import yaml
import json
import os
import logging
from shutil import copytree
from pprint import pprint

from funlib.geometry import Roi
from funlib.persistence import open_ds

from .volumes import make_volumes

this_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_and_update(config):
    logger.info(pprint(config))
    if click.confirm("Are these values good?", default=True):
        return config
    else:
        edited_yaml = click.edit(yaml.dump(config))
        if edited_yaml is not None:
            return yaml.safe_load(edited_yaml)
        else:
            return config


def save_config(config, filename):
    with open(filename, "w") as f:
        yaml.dump(config, f)
    logger.info(f"{filename} saved successfully.")


def copy_model_scripts(model_name, setup_dir):
    src = os.path.abspath(os.path.join(this_dir, "..", "models", model_name))
    logger.info(f"Copying {src} to {setup_dir}")
    copytree(src, setup_dir, dirs_exist_ok=True)


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


def choose_model(i, name):
    models = sorted(
        [
            d
            for d in os.listdir(os.path.join(os.path.dirname(__file__), "..", "models"))
            if os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "models", d))
            and "_from_" not in d
        ]
    )
    model = click.prompt(
        f"Enter model name for round {i+1}: {name}",
        type=click.Choice(models),
        default=models[0] if i == 0 else models[-1],
        show_choices=True,
        show_default=True,
    )

    return model


def create_training_config(round_dir, model_name, volumes):
    logger.info(f"\nTraining config for {round_dir}/{model_name}:")
    setup_dir = os.path.abspath(os.path.join(round_dir, model_name))
    copy_model_scripts(model_name, setup_dir)
    os.makedirs(os.path.join(setup_dir, "pipeline"), exist_ok=True)

    # get voxel size from volumes, assume all volumes have same voxel size
    voxel_size = volumes[0]["voxel_size"]

    sigma = None
    if "lsd" in model_name.lower():
        sigma = click.prompt(
            "Enter sigma for LSD model", default=10 * voxel_size[-1], type=int
        )

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

    if sigma is not None:
        train_config["sigma"] = sigma

    train_config = check_and_update(train_config)
    save_config(train_config, os.path.join(setup_dir, "pipeline", "train.yaml"))

    return train_config


def create_prediction_configs(volumes, train_config):

    setup_dir = train_config["setup_dir"]
    model_name = os.path.basename(setup_dir)
    round_name = os.path.basename(os.path.dirname(setup_dir))

    logger.info(f"\nPrediction configs for {round_name}/{model_name}:")

    pred_iter = click.prompt(
        "Enter checkpoint iteration for prediction",
        default=train_config["max_iterations"] - 1,
        type=int,
        show_default=True,
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
                this_dir, "..", f"models/3d_affs_from_{model_name.replace('_3ch','')}"
            )
        )
        affs_iter = click.prompt(
            f"Enter checkpoint iteration for round {round_name} inference of affs with {affs_setup_dir}",
            default=5000,
            type=int,
        )
        out_affs_ds = f"predictions/{round_name}-{model_name}/3d_affs_{affs_iter}_from_{pred_datasets[0].split('_')[-1]}"
    else:
        out_affs_ds = [x for x in pred_datasets if "3d_affs" in x][0]

    # can lsd errors be computed ?
    compute_lsd_errors = True if "3d_lsds" in model_outputs else False

    for volume in volumes:
        pred_config = {}
        volume_name = os.path.basename(volume["zarr_container"]).split(".zarr")[0]
        raw_array = os.path.join(volume["zarr_container"], volume["raw_dataset"])

        logger.info(f"\nCreating prediction config for {raw_array}:")

        roi_offset, roi_shape, _ = get_roi(in_array=raw_array)

        pred_config[model_name] = {
            "setup_dir": setup_dir,
            "raw_file": volume["zarr_container"],
            "raw_datasets": [
                volume["raw_dataset"],
            ],
            "roi_offset": list(roi_offset),
            "roi_shape": list(roi_shape),
            "checkpoint": os.path.join(setup_dir, f"model_checkpoint_{pred_iter}"),
            "out_file": volume["zarr_container"],
            "out_prefix": f"predictions/{round_name}-{model_name}",
            "num_workers": 4,
            "num_gpus": 4,
            "num_cache_workers": 1,
        }

        if make_affs:
            pred_config["affs"] = {
                "setup_dir": affs_setup_dir,
                "raw_file": volume["zarr_container"],
                "raw_datasets": pred_datasets,
                "roi_offset": list(roi_offset),
                "roi_shape": list(roi_shape),
                "checkpoint": os.path.join(
                    affs_setup_dir, f"model_checkpoint_{affs_iter}"
                ),
                "out_file": volume["zarr_container"],
                "out_prefix": f"predictions/{round_name}-{model_name}",
                "num_workers": 4,
                "num_gpus": 4,
                "num_cache_workers": 1,
            }

        save_config(
            check_and_update(pred_config),
            os.path.join(setup_dir, "pipeline", f"predict_{volume_name}.yaml"),
        )

    return out_affs_ds, pred_datasets, compute_lsd_errors, setup_dir


def create_segmentation_configs(volumes, out_affs_ds, setup_dir):

    round_name = os.path.basename(os.path.dirname(setup_dir))
    model_name = os.path.basename(setup_dir)
    logger.info(f"\nSegmentation configs for {round_name}/{model_name}:")

    waterz_defaults = {
        "fragments_in_xy": True,
        "background_mask": False,
        "mask_thresh": 0.5,
        "min_seed_distance": 10,
        "epsilon_agglomerate": 0.0,
        "filter_fragments": 0.05,
        "replace_sections": None,
        "thresholds_minmax": [0, 1],
        "thresholds_step": 0.05,
        "thresholds": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "merge_function": "mean",
    }

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

        # blockwise or not ?
        # do_blockwise = (
        #     roi_shape[0]
        #     * roi_shape[1]
        #     * roi_shape[2]
        #     / (voxel_size[0] * voxel_size[1] * voxel_size[2])
        #     > 536870912
        # )  # ~4GB, uint64

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

        seg_config = {
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

        save_config(
            check_and_update(seg_config),
            os.path.join(setup_dir, "pipeline", f"segment_{volume_name}.yaml"),
        )

    return out_seg_ds


def create_evaluation_configs(
    volumes, out_segs, out_lsd_errors, pred_datasets, setup_dir
):

    round_name = os.path.basename(os.path.dirname(setup_dir))
    model_name = os.path.basename(setup_dir)
    logger.info(f"\nEvaluation configs for {round_name}/{model_name}:")

    out_eval_dir = f"post/{round_name}-{model_name}/evaluations"
    out_map_ds = f"post/{round_name}-{model_name}/evaluations/lsd_error_map"
    out_mask_ds = f"post/{round_name}-{model_name}/evaluations/lsd_error_mask"

    for volume in volumes:
        volume_name = os.path.basename(volume["zarr_container"]).split(".zarr")[0]

        logger.info(f"\nCreating evaluation config for {out_segs}:")

        # ask if gt labels are available #TODO
        gt_labels_file = None
        gt_labels_dataset = None

        # ask if gt skeletons are available #TODO
        gt_skeletons_file = None

        # compute lsd errors ?
        if out_lsd_errors and click.confirm(
            f"Compute LSD errors for {out_segs}?", default=True, show_default=True
        ):
            lsds_file = volume["zarr_container"]
            lsds_dataset = [
                x for x in pred_datasets if "3d_lsds" in x and "_from_" not in x
            ][0]
            lsd_error_file = volume["zarr_container"]
            lsd_error_map_dataset = out_map_ds
            lsd_error_mask_dataset = out_mask_ds
        else:
            lsds_file = None
            lsds_dataset = None
            lsd_error_file = None
            lsd_error_map_dataset = None
            lsd_error_mask_dataset = None

        # add evaluation mask ? #TODO
        mask_file = None
        mask_dataset = None

        # get evaluation ROI TODO: get from segmentation config
        # roi_offset, roi_shape, voxel_size = get_roi(in_array=out_segs)

        eval_config = {
            "seg_file": volume["zarr_container"],
            "seg_datasets": out_segs,  # TODO: zarr tree find all seg arrays. eval on all.
            "out_dir": out_eval_dir,
            "mask_file": mask_file,
            "mask_dataset": mask_dataset,
            # "roi_offset": roi_offset,
            # "roi_shape": roi_shape,
        }

        if lsds_file is not None:
            eval_config["lsd_errors"] = {
                "lsds_file": lsds_file,
                "lsds_dataset": lsds_dataset,
                "out_file": lsd_error_file,
                "out_map_dataset": lsd_error_map_dataset,
                "out_mask_dataset": lsd_error_mask_dataset,
                "thresholds": [0.1, 1.0],
            }

        if gt_labels_file is not None or gt_skeletons_file is not None:
            eval_config["gt"] = {
                "gt_labels_file": gt_labels_file,
                "gt_labels_dataset": gt_labels_dataset,
                "gt_skeletons_file": gt_skeletons_file,
            }

        save_config(
            check_and_update(eval_config),
            os.path.join(setup_dir, "pipeline", f"evaluation_{volume_name}.yaml"),
        )

    return out_eval_dir


def create_filter_configs(
    volumes, out_segs, eval_dir, round_name, model_name, setup_dir
):

    logger.info(f"\nFilter configs for {round_name}/{model_name}:")

    out_seg_ds = f"pseudo_gt/{round_name}-{model_name}/ids"
    out_mask_ds = f"pseudo_gt/{round_name}-{model_name}/mask"

    out_volumes = []

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

        save_config(
            check_and_update(filter_config),
            os.path.join(setup_dir, "pipeline", f"filter_{volume_name}.yaml"),
        )

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

    return out_volumes


def make_round_configs(round_dir, model_name, volumes):
    """Create all configs for a model with given volumes."""

    train_config = create_training_config(round_dir, model_name, volumes)
    out_affs_ds, out_pred_datasets, out_lsd_errors, setup_dir = (
        create_prediction_configs(volumes, train_config)
    )
    out_segs = create_segmentation_configs(volumes, out_affs_ds, setup_dir)
    out_eval_dir = create_evaluation_configs(
        volumes, out_segs, out_lsd_errors, out_pred_datasets, setup_dir
    )
    out_volumes = create_filter_configs(
        volumes,
        out_segs,
        out_eval_dir,
        os.path.basename(round_dir),
        model_name,
        setup_dir,
    )

    return out_volumes


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


def make_configs(base_dir):
    """Create for multiple rounds with given volumes."""

    existing_rounds = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir((os.path.join(base_dir, d)))
        and ".zarr" not in d
        and "round" in d
    ]

    logger.info(f"Existing rounds: {existing_rounds}")

    num_rounds = click.prompt(
        "Enter number of bootstrapping rounds (more can be added later): ",
        type=int,
        default=1,
        show_default=True,
    )

    out_volumes = []

    for i in range(num_rounds):
        round_name = click.prompt(
            f"Enter name of round {i+1}: ", default=f"round_{i+1}"
        )
        round_dir = os.path.join(base_dir, round_name)
        os.makedirs(round_dir, exist_ok=True)

        # check if volumes.yaml exists
        if out_volumes == []:
            if not os.path.exists(os.path.join(round_dir, "volumes.yaml")):
                volumes = make_volumes(base_dir)
            else:
                with open(os.path.join(round_dir, "volumes.yaml")) as f:
                    volumes = yaml.safe_load(f)

                logger.info(f"Loaded volumes from {round_dir}/volumes.yaml: ")
                for volume in volumes:
                    logger.info(pprint(volume))

                if click.prompt(
                    "Do you want to update the volumes?",
                    default=False,
                    show_default=True,
                ):
                    volumes = make_volumes(base_dir)
                    save_config(volumes, os.path.join(round_dir, "volumes.yaml"))
        else:
            volumes = out_volumes

        model_name = choose_model(i, round_name)
        out_volumes = make_round_configs(round_dir, model_name, volumes)

        if i < num_rounds - 1:
            continue_round = click.prompt(
                "Continue to next round?",
                type=click.Choice(["y", "n"]),
                default="y",
            )
            if continue_round == "n":
                break
    logger.info(f"All configs created successfully!")
