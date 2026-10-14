# Changelog

## 1.0.0 (unreleased date)

### Breaking changes

- Predictions, fragments, segmentations and error maps go into the round, in a
  container named like the raw's, such as `round_01/data.zarr`. Your data
  containers stay as they are. To write next to the raw, set `outputs` in
  `volumes.toml`.
- `bs view PATH` takes array paths and opens a scale-pyramid group at its
  `s0`. Slices such as `raw[0:10]` are refused.
- An output key such as `seg_dataset_prefix` at step level is refused when the
  step runs two or more volumes. Set it in the table of each volume, such as
  `[segment.volume_2-zarr]`.
- Python 3.12 or later is necessary.
- Config files from 0.3.x do not load. Each step is now a table such as
  `[segment]` or `[train.<name>]`. The loader names the fix for an old key.
  Write a new round with `bs prepare round_01`.
- `bs prepare` takes the round as its argument: `bs prepare round_01`. The
  `-o` option and the round directory question are gone.
- `bs filter` is now `bs refine`, with the commands `size-filter`,
  `outlier-filter`, `z-filter`, `remap`, `morph` and `mask`.
- `bs utils clahe` is removed.
- Options use hyphens, not underscores: write `--voxel-size`, not
  `--voxel_size`.
- In `bs train`, `-s` means `--step`. `--save-snapshots-every` has no short
  flag.
- Watershed (`bs segment -ws`) needs the `waterz` extra:
  `uv pip install "bootstrapper[waterz] @ git+https://github.com/ucsdmanorlab/bootstrapper.git"`.
  It uses waterz 0.10 from PyPI, which never joins two objects through the
  background. Watershed segmentations change.
- A config file refuses every unknown key. The message names the table and the
  key, and gives the nearest valid name.
- `[segment.params]` takes only the params of the method that runs. Watershed
  params under `method = "mws"` are refused.
- `[segment.db]` has no `db_port`, `nodes_table` or `edges_table`. Set the
  port with `PGPORT` in the environment.
- `[evaluate]` has no `num_workers`. Error maps run in one process, so no tile
  is lost.
- A refine chain refuses `dry_run`. Preview one filter with
  `bs refine size-filter --dry-run`.
- In `volumes.toml`, `training_mask = "labels"` (sparsely labelled) or
  `"everywhere"` (densely labelled) replaces `sparse_labels`. `target_mask`
  replaces `mask`, and `mask_dilation = 10` replaces `dense_objects = false`.
  The loader names the replacement of each old key.
- Fragment filtering now runs. Fragments from 0.3.x were never filtered,
  although their names say they were. Make them again before you compare or
  agglomerate them.
- Watershed and mutex watershed always run as volara tasks. With no
  `[segment.db]`, or an empty one, they write a SQLite database beside the
  output.
- Error maps are scaled by the channel count, not by the maximum of each block.
  So a value means the same in every block and every run. An old value is the
  new value times 255 divided by the new maximum of its block. A threshold
  tuned on old maps selects less now.
- The `3d_affs` and `3d_mtlsd` models write nine affinity channels, one for
  each offset of their neighborhood. A checkpoint with a six-channel head does
  not load in a setup that `bs prepare` writes now.
- Training changes (see Fixed), so a model trained with 1.0 differs from a
  0.3.x model. The published checkpoints come from the old code and stay as
  they are.
- Prediction with `num_workers` above 1 always runs daisy blocks on the output
  chunk grid, also on one GPU. Before, two workers on one GPU could write one
  chunk and zero part of it. Make again any prediction made with
  `num_gpus = 1` and `num_workers` above 1.

### Added

- `bs prepare round_01 --raw ...` on an existing round adds a volume. It writes
  the step files the round now needs, such as an evaluate step for a volume
  with ground truth.
- `bs run round_01` starts at the first step that is not done or that changed.
  After an edit in `04_segment.toml`, it runs steps 04 to 06 only.
  `--from 01` runs every step.
- `bs view round_01` opens the raw and every array the round wrote, with short
  layer names such as `03 3d_affs (z y x)`. `bs view round_01 04` opens one
  step. After each step, `bs run` prints its `bs view` command.
- `bs prepare` downloads the pretrained checkpoints that the round needs.
- `bs prepare round_01 --raw data.zarr/raw/s0 --labels labels/s0 --labelled sparsely`
  writes a round with no questions. Every question has a flag, and
  `bs prepare --help` gives an example of each.
- With no terminal, `bs prepare` never asks. A missing fact stops it with its
  flag, its choices and an example.
- The `bs prepare` dialogue lists the arrays it finds and takes a number or a
  path. It checks each array against the voxel size of the raw. `?` explains a
  question, `<` goes back, and Tab completes paths.
- `bs prepare round_01.toml` writes the whole round in one file.
- `bs prepare` asks for models by name or short form, such as `2M 3Af2L`. `?`
  lists the 9 chains that make one round. Another chain is refused with the
  fix.
- `bs utils download-ckpts` takes every short form, such as `3Af2L`. It
  refuses a model that trains on your labels.
- `bs prepare` writes commented config files. A comment says what each param
  does, and optional keys show as commented examples.
- Each config file holds your keys, then a run tail of derived keys in
  `[run.*]` tables. After you edit a key, `bs prepare round_01` writes the
  tails again. It keeps your lines and prints the keys that changed.
- `bs run round_01 --dry-run` checks every path and prints what each step
  reads and writes. It runs nothing.
- `bs run --only` and `bs run --from` run a part of the steps.
- Each `bs run` writes a receipt, `runs/<stamp>.toml`, with the result, the
  time and the keys of each step. `bs run` on a receipt repeats the steps.
- A volume path can be a glob. Each match gives one volume, named by its
  container: `crop1.zarr` gives `crop1-zarr`.
- A volume can name arrays of its own, such as `pgt = ".../pgt/s0"`.
- The train step key `sampling` sets how often each volume is drawn. It takes
  `"labels"` (default), `"equal"` or `"inv"`. It also takes numbers, as a list
  such as `[3, 1]` or a table such as `{a-zarr = 3, b-zarr = 1}`. `bs train`
  prints the draw.
- User seeds for mutex watershed. Set `seeds_dataset` in `[segment]` to an
  integer labels array on the affs voxel grid. Two seed ids never merge, also
  across blocks. Unseeded objects get ids above the largest seed id. Seeded
  output names start with `seeded--`. A seed id must be 0 to 2^63 - 1.
- Quantile merge functions for watershed: `hist_quant_10`, `hist_quant_25`,
  `hist_quant_50`, `hist_quant_75` and `hist_quant_90`, each also with
  `_initmax`. The default stays `mean`.
- `bs segment` prints the params that took a default, such as
  `defaults applied: bias, sigma=None, noise_eps=0.001`.
- `bs refine morph --op` takes `fill-holes`, `dilate`, `erode`, `opening` and
  `closing`. Closing gives each filled voxel the label of the nearest
  labelled voxel. So it does not erode the contact between two labels.
- `bs refine morph --op fill-holes` fills background enclosed by one object
  with the id of that object. An enclosed object stays, unless it has at most
  `--max-area` voxels (default 0).
- `bs refine morph --exclude-ids 1,4,8` (`exclude_ids = [1, 4, 8]` in a config
  file) keeps dilate and fill-holes away from these ids.
- `bs refine mask` sets labels outside a mask to 0.
- `bs utils tissue-mask RAW OUT` writes a tissue mask with no holes. It fills
  black sections and folds of up to 4 sections from the sections around them.
  It closes gaps narrower than `--radius` voxels (default 24). It works on
  blocks of whole sections, so the stack does not have to fit in memory.
- `bs utils scale-pyramid --labels` or `--image` sets how the array is
  resampled.
- `bs evaluate` reports each segmentation. The report gives the object count,
  the foreground fraction, the voxels outside the mask and the 3 largest ids. A step with no
  `gt` and no `pred` table runs only the report.
- `bs view --bind` sets the address of the viewer.
- `bs --version` prints the version.
- A crash prints one line, and the traceback goes to `bs_error_<stamp>.log`.
- A `Dockerfile` builds an image: `docker build -t bootstrapper:1.0.0 .`

### Changed

- Segment blocks are at least 16 sections deep by default, so a 3D model's
  4-section chunks no longer put a block face every 4 sections.
- `bs view` shows affinities and LSDs as RGB of channels 0 to 2, and ids and
  masks as segmentation layers. It opens on the centre of the volume with
  ground truth, with a section in the main panel.
- `bs evaluate` ends with one line per volume and segmentation, such as
  `volume_2-zarr  mws/fill_holes  NVI split 0.115 merge 0.145`.
- Each blockwise task prints one line when it ends. A failed block still
  prints the full daisy summary. A CPU smoke round prints 73 lines in place of
  257.
- Training prints its iterations, data workers and GPUs before it starts. It
  sets the float32 matmul precision to "high".
- `bs run --only` and `--from` take a step number, such as `04`.
- The labels question has three answers: sparsely labelled, densely
  labelled, or densely labelled within a mask. `bs prepare` finds 2D labels
  itself and prints why. Before it calls the labels 2D, it reads every
  section, so it also finds labels between the sampled sections.
- Labels on one section, or on every second section, count as 2D. The default
  models are then `2d_mtlsd` and `3d_affs_from_2d_mtlsd`.
- The default volume name is the container file name with the dot as a dash:
  `data.zarr` gives `data-zarr`.
- `bs prepare` writes each training mask into the round, at
  `round_01/training_mask.zarr/<volume>`, and nothing into your data.
- `bs prepare` copies the model scripts only into a setup with no `train.py`,
  so your edits to `net_config.json` stay.
- `bs prepare` does not ask for GPUs. Each predict file holds `num_gpus = 1`.
- The thresholds of the prediction check are in `[evaluate]`, once for all
  volumes.
- With a target mask, predict covers only the box of the mask. Refine ends with
  a mask op that sets labels outside the mask to 0. One volume with a target
  mask is enough. `bs prepare` also adds the op when it adds a volume with
  `--target-mask` to a round.
- With several volumes, training draws each volume by its labelled voxels.
  Before, each volume was drawn equally often. `sampling = "equal"` restores
  that.
- Inside a volume, training draws each run of adjacent labelled sections by
  its labelled voxels. So a section with a small labelled patch is drawn less
  often than a fully labelled section.
- `bs train` reads each training mask once before training, to find the
  labelled sections. On a 200-section volume this takes 0.35 s.
- The 2D LSD sigma is 10 voxels at all voxel sizes (`"sigma": 10` in
  `net_config.json`). Before, it was 80 nm: 0.48 voxels at 168 nm and 80
  voxels at 1 nm. At 8 nm voxels nothing changes.
- The tables of a volume merge key by key over the tables of the step.
- `num_workers` in `[refine]` is the default of each op, and an op table can
  set its own.
- `bs evaluate` runs the `gt` and `pred` checks of each volume table. Each
  volume writes its own result file, such as `06_evaluate_apical_gt.json`.
- `bs run` refuses a round whose run tails are older than your keys. The error
  names the changed keys and the fix, `bs prepare round_01`.
- `bs run` stops before the first step when an input path is missing. An
  input that an earlier step writes is not missing.
- `bs train` stops before training when the labels of a sample reach past its
  raw. It also stops when the mask reaches past the labels. The message gives
  the crop to use.
- `bs prepare` stops when a training mask does not overlap the labels. The
  message names a mask that overlaps, if one exists.
- `bs train` names the `bs prepare` command to run when the setup has no
  `train.py` or a training mask is missing.
- `num_gpus = 0` runs prediction on the CPU, also when `CUDA_VISIBLE_DEVICES`
  is set but empty.
- One prediction worker runs on the first granted GPU. If
  `CUDA_VISIBLE_DEVICES` is not set, `bs predict` warns that it uses GPU 0.
- A prediction ROI smaller than one output block is refused with a message.
- `bs predict` refuses an existing output array with another shape, offset,
  voxel size or dtype. Remove it, or set another `output_datasets_prefix`.
- Mutex watershed cuts the segment ROI to the affs and snaps it to the affs
  voxel grid. It prints one line when the ROI changes.
- Each segmentation output prefix has its own volara log dir,
  `<container>_volara_logs/<prefix>`.
- Commands print fewer INFO log lines.
- Training uses half of the CPUs as data workers, at most 32, in place of 8.
  On one RTX 4090, `2d_mtlsd` trains 3.6 times faster.
- Training uses 1 GPU by default. Set `devices = 2` or `devices = [0, 1]` in
  the train file for more.
- `bs prepare` writes each training mask from the current labels every time.

### Fixed

- `method = "cc"` read the affinities one voxel off. Boundaries moved by one voxel, and
  touching objects joined. It now reads them as bs trains them.
- A `*` that matches more than one array stops `bs prepare` with the fix:
  give one array in full.
- `bs predict` asks no question. A missing checkpoint stops `bs run` and its
  dry run before the first step, with the command that downloads it.
- `bs run` predicts with the last checkpoint that training saves. With
  `max_iterations = 12001`, this is `model_checkpoint_10000`.
- `bs prepare` stops when `max_iterations` is under `save_checkpoints_every`,
  because training then saves no checkpoint. Example: `max_iterations = 5001`.
- `bs prepare` on a round directory with only `volumes.toml` uses the
  `models` key of that file.
- A refine mask op runs only on the volumes that have a target mask.
- `bs prepare ... | head` ends quietly, with no traceback file.
- The 3D trainers (`3d_affs`, `3d_lsd`, `3d_mtlsd`) deform each section in y
  and x only. Before, one output section could mix up to 18 source sections.
- The six trainers that read your data keep raw on its labels. Before, raw
  could be up to 14 voxels off its labels.
- The same six trainers take only batches whose output window is more than
  5 % labelled. Before, a batch could reach the loss with no labelled voxel.
- The gamma augmentation changes the image. The defect augmentation uses its
  deformation probability.
- A raw with a channel axis (c z y x) trains and predicts with the 6 models
  that read raw. Before, training stopped with "array shift axis length ...
  is not divisible" or a conv3d input error. `bs prepare` writes the channel
  count as `in_channels` in `net_config.json`. All raws of one round need the
  same channel count. The defect augmentation changes whole sections of such
  a raw. Shift augment is off for it.
- Training draws its windows only on the sections that hold labels. Before,
  it drew over the whole volume and rejected the windows without labels. With
  labels on 1 of 30 sections, a window took 25 draws. With labels on every
  21st of 200 sections, it took 20. Now each window takes 1 draw. On one CPU
  process, a batch of 10 windows from the 30-section volume takes 0.75 s, not
  11.3 s.
- Densely labelled volumes also train on the space between objects.
  `mask_dilation` grows the training mask of sparsely labelled volumes.
  `training_mask = "<path>"` trains on that mask.
- Each model in a chain gets only the arrays it reads. Before,
  `3d_affs_from_2d_lsd` and `3d_affs_from_2d_affs` got both outputs of
  `2d_mtlsd`, and predict stopped.
- Prediction with several workers on one GPU does not hang
  after a worker stops.
- A prediction worker that stops also stops its model process, so no process
  keeps a GPU after `bs` ends.
- `bs evaluate` error maps keep every tile. They work on an ROI smaller than
  one prediction chunk, and LSD error maps work for any sigma.
- `bs evaluate` refuses a mask on another voxel grid than the segmentation.
- `bs utils bbox` and `bs utils convert --crop` refuse a volume with no
  foreground and name the file.
- `bs utils scale-pyramid -i` on a pyramid group names the level to give, such
  as `-i pyr.zarr/raw/s1`.
- A zarr container given as the raw image does not crash the converter.
- `bs view --snapshot` works when the snapshot container holds a group.
- `bs view` sets the contrast of a raw with channels from its data. Channels 0
  to 2 each map their 1st to 99.5th percentile on the centre section to black
  and white. Before, a dim raw showed near black.
- A model copy from `bs prepare` or `bs utils download-ckpts` has no
  `__pycache__`.
- A setup `train.py` run with the config of another setup names both
  directories.
- A failed checkpoint download stops with a message that names the URL.
- `bs evaluate` stops with an error when no segmentation exists under
  `seg_datasets_prefix`.
