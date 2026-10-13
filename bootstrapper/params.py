"""The keys bs prepare writes into the config files: each one's value and what it does.
A table name has * for a model or volume name. An off key shows as a commented example."""
from dataclasses import dataclass

from .configs import EVAL_ERROR_THRESHOLDS
from .segment import DEFAULTS


@dataclass(frozen=True)
class Key:
    value: object
    help: str
    off: bool = False


MWS = DEFAULTS["mws"]
KEYS = {
    "volumes.*": {
        "training_mask": Key(None, 'Sparsely labelled: "labels"; densely labelled: "everywhere"; densely labelled within a mask: its path.'),
        "2d_labels": Key(None, "true: the labels sit on single sections."),
        "mask_dilation": Key(None, "Trains on this many voxels of background around each object."),
        "target_mask": Key(None, "Segment, refine and evaluate only inside this mask; predict covers its box."),
        "gt": Key(None, "Ground truth for evaluate."),
        "outputs": Key(None, "Where outputs go. Default: the raw's container."),
        "voxel_size": Key(None, "Voxel size in nm, z y x."),
    },
    "train.*": {
        "max_iterations": Key(30001, "Training stops after this many iterations; 30001 saves a last checkpoint at 30000."),
        "save_checkpoints_every": Key(5000, "Iterations between checkpoints."),
        "save_snapshots_every": Key(1000, "Iterations between snapshots: training batches you can view."),
        "devices": Key("auto", "GPUs for training: auto, or a list such as [0]."),
    },
    "predict.*": {
        "num_gpus": Key(1, "GPUs for prediction. More than one runs blocks through daisy."),
        "num_workers": Key(1, "Prediction workers. More than one runs blocks through daisy."),
    },
    "segment": {
        "method": Key("mws", "mws (mutex watershed), ws (watershed, needs the waterz extra) or cc (connected components)."),
        "blockwise": Key(True, "Segments in blocks with several workers. false segments the whole array as one block."),
        "num_workers": Key(10, "Block workers. SQLite locks above 10; use PostgreSQL in [segment.db] for more."),
        "block_shape": Key([32, 512, 512], "Block shape in voxels, z y x. Default: the affinities' chunks.", off=True),
    },
    "segment.params": {
        "bias": Key(MWS["bias"], "Added to each affinity channel, one value per offset. More negative splits more."),
        "global_bias": Key(MWS["global_bias"], "Weight and bias: two fragments join when weight x mean affinity + bias > 0."),
        "noise_eps": Key(MWS["noise_eps"], "Noise on the affinities. It breaks ties and stops streaks."),
        "filter_fragments": Key(MWS["filter_fragments"], "Removes fragments whose mean affinity is below this value."),
        "remove_debris": Key(MWS["remove_debris"], "Removes fragments smaller than this many voxels."),
    },
    "segment.db": {
        "db_host": Key("localhost", "Empty: an SQLite file beside the segmentations. For more than 10 workers, PostgreSQL:", off=True),
        "db_name": Key("bootstrapper", "", off=True),
        "db_user": Key("postgres", "", off=True),
        "db_password": Key("...", "", off=True),
    },
    "refine.fill-holes": {
        "xy": Key(True, "Fills holes section by section; false fills them in 3D."),
    },
    "evaluate": {
        "thresholds": Key(EVAL_ERROR_THRESHOLDS["affs"], "The error mask keeps affinity errors between these two values."),
    },
}

TAIL = {  # what a derived key of the run tail holds
    "setup_dir": "The model scripts and checkpoints.",
    "voxel_size": "Voxel size in nm, z y x, from the raw.",
    "checkpoint": "The last checkpoint of the train step, or the shipped one.",
    "input_datasets": "The raw, or what the predict step before gives.",
    "roi_offset": "With roi_shape, the target mask's box in nm: predict covers only it.",
    "aff_neighborhood": "The affinity offsets in the model's net_config.json.",
    "mask_dataset": "The target mask.",
    "mask_array": "The target mask.",
    "in_array": "The segmentation that the segment step writes with its params.",
}


def defaults(table):
    """The values of a table's keys that are on."""
    return {k: key.value for k, key in KEYS[table].items() if not key.off}
