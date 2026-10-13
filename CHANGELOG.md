# Changelog

## Unreleased

### Breaking changes

- Fragment filtering now runs. Fragments made before this release were never
  filtered, although their names and their recorded params say they were.
  Make them again with the fixed code before you compare or agglomerate them.
- Training augmentation is fixed. A model trained after this release differs
  from a model trained before it. The published checkpoints come from the old
  code and stay as they are.
- The in-memory watershed and mutex watershed paths (`simple_watershed`,
  `simple_mutex`) are deleted. Both methods now always run as volara tasks, so
  a config with no `db` gets a SQLite database beside the output. A whole-array
  run is the one-block case. Connected components keeps its whole-array path.
- Error maps are scaled by the channel count, not by each block's own maximum.
  A value is now absolute, so it means the same thing in every block and in
  every run. Maps written before this release are block-relative and cannot be
  compared with new ones: an old value is the new value times 255 divided by
  that block's new maximum. A threshold tuned on the old maps selects less now.
- The `3d_affs` and `3d_mtlsd` models declare nine affinity channels, matching
  their nine-offset neighborhood, where they declared six. Prediction sizes its
  output from the same value, so those models now write nine channels. A
  checkpoint trained with a six-wide head does not load under a setup directory
  prepared after this release.
