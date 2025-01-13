# CREMI example

Here we will bootstrap a segmentation of a CREMI volume using sparse labels on a single section.

## Download
```
python download.py
```
This downloads CREMI sample C and creates a zarr container `cremi_c.zarr` with the raw data, gt labels, and sparse labels.

```
cremi_c.zarr
 ├── gt_labels (125, 1250, 1250) uint64
 ├── raw (125, 1250, 1250) uint8
 └── sparse_labels (62, 1250, 1250) uint64
```

In general it is always a good idea to view and inspect the data before training, this can be done with:
```
bs view cremi_c.zarr/*
```


Since the zarr is already made, we do not need to convert volumes when we use `bs prepare`. 

## Setup 

* `bs prepare`

This interactive command will:
- Create a directory structure for the round
- Set up models
- Generate necessary config files for the pipeline

The command creates a `round_1` directory with the following structure:
```
round_1/
├── setup_01/                        # model directory (where checkpoints, logs, and snapshots are saved)
│   ├── net_config.json                # model config parameters (number of feature maps, kernel sizes, etc.)
│   ├── model.py                       # Pytorch module and loss definitions
│   ├── unet.py                        # unet implementation
│   ├── train.py                       # training script
│   └── predict.py                     # prediction script
└── run/                             # round config files directory
    ├── 01_train_00.toml              # training config
    ├── 02_pred_volume_1.toml         # prediction config
    ├── 03_seg_volume_1.toml          # segmentation config
    ├── 04_eval_volume_1.toml         # evaluation config
    └── 05_filter_volume_1.toml       # filter config
```

## Workflow
### 1. Training
Train the model using:
* `bs train round_1/run/01_train_00.toml`


Monitor training progress by viewing snapshots:

* `bs view -s round_1/setup_01/snapshots/batch_1.zarr`

### 2. Prediction
Generate predictions using:

* `bs predict round_1/run/02_pred_volume_1.toml`

View the predictions:

* `bs view round_1/volume_1.zarr/3Af2M/3000--from--setup_01_5000/3d_affs`

### 3. Segmentation
Run segmentation on the predictions:

* `bs segment round_1/run/03_seg_volume_1.toml`

View segmentation results:

* `bs view round_1/volume_1.zarr/3Af2M/3000--from--setup_01_5000/segmentations*/*`

### 4. Evaluation
Evaluate the segmentation outputs:
* `bs eval round_1/run/04_eval_volume_1.toml`


### 5. Filtering
Apply optional filtering to the segmentation outputs:
* `bs filter round_1/run/05_filter_volume_1.toml`


## Next Round
To configure the next bootstrapping round:
* `bs prepare`


## Notes
- Each volume gets dedicated config files for prediction, segmentation, evaluation, and filtering
- All pipeline configs are stored in the `run` directory