# CREMI example

Here we will bootstrap a segmentation of a CREMI volume using sparse labels on a single section.

## Installation
* Rust is necessary if you wish to use `mwatershed` for segmentation. Install from [rustup.rs](https://rustup.rs/)
* `git clone https://github.com/ucsdmanorlab/bootstrapper.git`
* `cd bootstrapper` -> `pip install .[all]`

The rest of the example is relative from `examples/cremi`.

## Data Download
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

For this example, answer the prompts as follows:
<details>
<summary> Show example prompt responses </summary>

1. **Initial Setup**:
    - Accept default base directory by pressing Enter
    - Accept default round name `round_1` by pressing Enter
    - Accept default number of volumes (`1`) by pressing Enter
    - Accept default volume name `volume_1` by pressing Enter

2. **Input Data (volume 1)**:
    - For *RAW* input: enter `cremi_c.zarr/raw`
        - Answer "`N`" for bounding box crop
        - Answer "`N`" for copying to output container
        - Answer "`N`" for raw mask
    - For *LABELS* input: enter `cremi_c.zarr/sparse_labels`
        - Answer "`Y`" for bounding box crop
        - Answer "`N`" for copying to output container
        - Accept default labels dataset path by pressing Enter
        - Answer "`N`" for labels mask
    - Accept volume settings by answering "`N`" to edit
3. **Model and Training Configuration**:
    - Accept default run directory by pressing Enter
    - For model name enter: `2d_affs`
    - Answer "`Y`" to add `3d_affs_from_2d_affs`
        - Accept default setup directory by pressing Enter
        - Answer "`N`" to edit net_config.json
        - Choose "`pretrained`" for `3d_affs_from_2d_affs`; Accept default directory for `3d_affs_from_2d_affs`
        - Answer "`Y`" to download pretrained checkpoints
    - Training Parameters for `2d_affs`:
        - Accept default max iterations (`30001`)
        - Accept default checkpoint save frequency (`5000`)
        - Accept default snapshot frequency (`1000`)
        - Answer "`N`" to edit configuration
4. **Prediction Configuration**:
    - Enter `20000` for `2d_affs` checkpoint iteration
    - Enter `5000` for `3d_affs_from_2d_affs` checkpoint itteration
    - Accept defaults for GPUs and CPU workers
    - Accept default voxel offset and shape
    - Answer "`N`" to edit configuration
5. **Segmentation Configuration**:
    - Choose "`mws`" as segmentation method
    - Accept default MWS parameters
    - Answer "`N`" to blockwise
    - Answer "`N`" to edit configuration
6. **Evaluation**:
    - Answer "`Y`" for ground truth labels
    - Enter `cremi_c.zarr/gt_labels` for ground truth path
    - Answer "`N`" for skeletons
    - Accept default for prediction errors
    - Answer "`N`" to edit configuration
7. **Filtering**:
    - Accept default filter settings
    - Answer "`N`" to edit configuration
    - Answer "`N`" to make configs for next round

</details>

-------------------------------------------

This will create a `round_1` directory with the following structure:
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