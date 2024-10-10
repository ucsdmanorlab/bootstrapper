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
 └── sparse_labels (1, 1250, 1250) uint64
```

In general it is always a good idea to view the data before training, this can be done with:
```
bs view cremi_c.zarr/*
```

## Prepare configs

Since the zarr is already made, we do not need to convert volumes when we use `bs prepare`.

* `bs prepare`
```
>>> base dir : .
>>> raw array: cremi_c.zarr/raw
>>> labels array: cremi_c.zarr/sparse_labels
>>> round name : round_1
>>> round_1 model: 2d_mtlsd_3ch
>>> iters: 5000
>>> gt labels: cremi_c.zarr/gt_labels
>>> gt skeletons: cremi_c_gt_skeletons.graphml
>>> compute prediction errors: True
>>> compute gt errors: True
```

This creates a `round_1` directory in the base directory with the following structure:
```
round_1                                 # round directory
└── 2d_mtlsd_3ch                        # model setup directory
    ├── net_config.json                 # model config parameters
    ├── model.py                        # model and loss definitions
    ├── unet.py                         # necessary modules for model
    ├── train.py                        # training script
    ├── predict.py                      # prediction script
    └── pipeline                        # contains config files
        ├── train.yaml                  # training config
        ├── predict_cremi_c.yaml        # prediction config
        ├── segment_cremi_c.yaml        # segmentation config
        ├── evaluate_cremi_c.yaml       # evaluation config
        └── filter_cremi_c.yaml         # filter config
```

* Every volume gets its own config file for prediction, segmentation, evaluation, and filtering.
* All config files for the model are inside the `pipeline` directory.

## Train
- ```bs train round_1/2d_mtlsd_3ch/pipeline/train.yaml```
- View snapshots with `bs view -s round_1/2d_mtlsd_3ch/snapshots/batch_1.zarr`

## Predict
- `bs predict round_1/2d_mtlsd_3ch/pipeline/predict_cremi_c.yaml`
- `bs predict round_1/2d_mtlsd_3ch/pipeline/predict_cremi_c.yaml affs`
- `bs view cremi_c.zarr/predictions/round_1-2d_mtlsd_3ch/*`

## Segment
- `bs segment round_1/2d_mtlsd_3ch/pipeline/segment_cremi_c.yaml`
- `bs view cremi_c.zarr/post/round_1-2d_mtlsd_3ch/segmentations/*/*`

## Evaluate
- `bs eval round_1/2d_mtlsd_3ch/pipeline/evaluate_cremi_c.yaml`

## Filter
- `bs filter round_1/2d_mtlsd_3ch/pipeline/filter_cremi_c.yaml`

## Next round
* `bs prepare`
```
>>> base dir : .
>>> raw array: cremi_c.zarr/raw
>>> labels array: cremi_c.zarr/pseudo_gt/round_1-2d_mtlsd_3ch/ids
>>> round name : round_2
>>> round_1 model: 3d_mtlsd
>>> iters: 50000
>>> gt labels: cremi_c.zarr/gt_labels
>>> gt skeletons: cremi_c_gt_skeletons.graphml
>>> compute prediction errors: True
>>> compute gt errors: True
```