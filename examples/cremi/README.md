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

### **For this example, answer the prompts as follows (click to expand)**

1.  <details><summary> Initial setup </summary>

    - Accept default base directory by pressing Enter
    - Accept default round name `round_1` by pressing Enter
    - Accept default number of volumes (`1`) by pressing Enter
    - Accept default volume name `volume_1` by pressing Enter

    </details>
2. <details><summary> Processing volume 1 </summary>

    - For *RAW* input: enter `cremi_c.zarr/raw`
        - Answer "`N`" for bounding box crop
        - Answer "`N`" for copying to output container
        - Answer "`N`" for raw mask (Answer "`Y`" if your image data has black / background borders that can be ignored during post-processing)
    - For *LABELS* input: enter `cremi_c.zarr/sparse_labels`
        - Answer "`Y`" for bounding box crop (This greatly speeds up training if your labels array has a lot of empty space)
        - Answer "`N`" for copying to output container
        - Accept default labels dataset path by pressing Enter
        - Answer "`Y`" for make/provide labels mask. Answer "Y" again to make. (Always try to explicity provide training masks to avoid generating on the fly)
    - Accept volume settings by answering "`N`" to edit. You should have:
    -   <details>
        <summary> Show volume config </summary>

        ```
        {'volume_1': {'labels_dataset': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/labels',
                    'labels_mask_dataset': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/labels_mask',
                    'name': 'volume_1',
                    'output_container': 'bootstrapper/examples/cremi/round_1/volume_1.zarr',
                    'raw_dataset': 'bootstrapper/examples/cremi/cremi_c.zarr/raw',
                    'raw_mask_dataset': None,
                    'voxel_size': [40, 4, 4]}}
        ```
        
        </details>
   </details> 
3. <details><summary> Creating training configs </summary>

    - Accept default run directory by pressing Enter
    - For model name enter: `2d_lsd`
    - Answer "`Y`" to add `3d_affs_from_2d_lsd` to training config.
        - Accept default setup directory by pressing Enter
        - Answer "`N`" to edit net_config.json
        - Choose "`pretrained`" for `3d_affs_from_2d_lsd`; Accept default directory for `3d_affs_from_2d_lsd`
        - Answer "`Y`" to download pretrained checkpoints
    - Training Parameters for `2d_lsd`:
        - Accept default max iterations (`30001`)
        - Accept default checkpoint save frequency (`5000`)
        - Accept default snapshot frequency (`1000`)
        - Answer "`N`" to edit configuration. You should have:
    -   <details>
        <summary> Show training config </summary>

        ```
        {'max_iterations': 30001,
        'samples': [{'labels': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/labels',
                    'mask': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/labels_mask',
                    'raw': 'bootstrapper/examples/cremi/cremi_c.zarr/raw'}],
        'save_checkpoints_every': 5000,
        'save_snapshots_every': 1000,
        'setup_dir': '/home/vijay/bootstrapper/examples/cremi/round_1/setup_01',
        'voxel_size': [40, 4, 4]}
        ```

        </details>
   </details> 
4. <details><summary> Creating prediction configs </summary>

    - Enter `20000` for `2d_lsd` checkpoint iteration
    - Enter `15000` for `3d_affs_from_2d_lsd` checkpoint itteration
    - Accept defaults for GPUs and CPU workers
    - Accept default voxel offset and shape
    - Answer "`N`" to edit configuration. You should have:
    -   <details>
        <summary> Show prediction config </summary>

        ```
        {'01-setup_01': {'chain_str': '',
                    'checkpoint': 'bootstrapper/examples/cremi/round_1/setup_01/model_checkpoint_20000',
                    'input_datasets': ['bootstrapper/examples/cremi/cremi_c.zarr/raw'],
                    'num_gpus': 1,
                    'num_workers': 1,
                    'output_datasets_prefix': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/setup_01',
                    'roi_offset': [0, 0, 0],
                    'roi_shape': [5000, 5000, 5000],
                    'setup_dir': 'bootstrapper/examples/cremi/round_1/setup_01'},
        '02-3Af2A': {'chain_str': 'setup_01_20000',
                    'checkpoint': 'bootstrapper/bootstrapper/models/3d_affs_from_2d_lsd/model_checkpoint_15000',
                    'input_datasets': ['bootstrapper/examples/cremi/round_1/volume_1.zarr/setup_01/20000/2d_lsds'],
                    'num_gpus': 1,
                    'num_workers': 1,
                    'output_datasets_prefix': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/3Af2L',
                    'roi_offset': [0, 0, 0],
                    'roi_shape': [5000, 5000, 5000],
                    'setup_dir': 'bootstrapper/bootstrapper/models/3d_affs_from_2d_lsd'}}
        ```

        </details>
   </details> 
5. <details><summary> Creating segmentation configs </summary>

    - Choose "`ws`" as segmentation method
    - Accept default watershed parameters
    - Answer "`N`" to blockwise
    - Answer "`N`" to edit configuration. You should have:
    -   <details>
        <summary> Show segmentation config </summary>

        ```
        {'affs_dataset': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/3Af2L/15000--from--setup_01_20000/3d_affs',
        'block_shape': None,
        'blockwise': False,
        'context': None,
        'fragments_dataset': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/3Af2L/15000--from--setup_01_20000/fragments_ws',
        'lut_dir': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/3Af2L/15000--from--setup_01_20000/luts_ws',
        'mask_dataset': None,
        'ws_params': {
            'fragments_in_xy': True,
            'min_seed_distance': 10,
            'epsilon_agglomerate': 0.0,
            'filter_fragments': 0.05,
            'thresholds_minmax': [ 0, 1,],
            'thresholds_step': 0.05,
            'thresholds': [ 0.2, 0.35, 0.5,],
            'merge_function': "mean"},
        'num_workers': 1,
        'seg_dataset_prefix': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/3Af2L/15000--from--setup_01_20000/segmentations_ws'}
        ```

        </details>
   </details> 
6. <details><summary> Creating evaluation configs </summary>

    - Answer "`Y`" for ground truth labels
    - Enter `cremi_c.zarr/gt_labels` for ground truth path
    - Answer "`N`" for skeletons
    - Accept default for prediction errors
    - Answer "`N`" to edit configuration. You should have:
    -   <details>
        <summary> Show evaluation config </summary>

        ```
        {'gt': {'labels_dataset': '/home/vijay/bootstrapper/examples/cremi/cremi_c.zarr/gt_labels',
            'skeletons_file': None},
        'mask_dataset': None,
        'out_result_dir': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/3Af2L/15000--from--setup_01_20000',
        'pred': {'params': {'aff_neighborhood': [[-1, 0, 0],
                                                [0, -1, 0],
                                                [0, 0, -1],
                                                [-2, 0, 0],
                                                [0, -8, 0],
                                                [0, 0, -8]]},
                'pred_dataset': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/3Af2L/15000--from--setup_01_20000/3d_affs',
                'thresholds': [0.1, 1.0]},
        'seg_datasets_prefix': 'bootstrapper/examples/cremi/round_1/volume_1.zarr/3Af2L/15000--from--setup_01_20000/segmentations_ws'}
        ```

        </details>
   </details> 
7. <details><summary> Creating filter configs </summary>

    - Accept default filter settings
    - Answer "`N`" to edit configuration
    - Answer "`N`" to make configs for next round

-------------------------------------------

This will create a `round_1` directory with the following structure:
```
round_1/
├── volumes.toml                     # volume config file (for easier preparation of new setups)
├── volume_1.zarr                    # output container for volume 1 (also contains the bounding boxed labels and mask)
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

* `bs view round_1/volume_1.zarr/3Af2L/15000--from--setup_01_20000/3d_affs`

### 3. Segmentation
Run segmentation on the predictions:

* `bs segment round_1/run/03_seg_volume_1.toml`

View segmentation results:

* `bs view round_1/volume_1.zarr/3Af2L/15000--from--setup_01_20000/segmentations*/*`

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