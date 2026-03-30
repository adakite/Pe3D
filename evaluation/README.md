# Evaluation Overview

This directory contains validation and benchmark outputs for the Python `pe3d` fork against the bundled MATLAB reference workflow and against manually labeled datasets.

The most useful subdirectories for a public repository are:

- `pe3d_current_compare_normalized/`: current MATLAB-vs-`pe3d` Otira comparison after normalizing angle conventions to `[0, 180)`
- `preprocess_mode_benchmark/`: timing and parity comparison for the `matlab` and `fast` preprocessing modes
- `worker_benchmark/`: runtime and output-parity check for serial versus auto-threaded grain fitting
- `orientation_focus_compare/`: focused orientation comparison snapshots
- `manual_label_block_tuning/`: tuning notes for manual block/non-block evaluation

## Recommended interpretation

- Size metrics are the strongest part of the current Python port.
- Orientation remains the main area where the Python fork is less faithful to the MATLAB reference.
- Benchmarks should be presented as validation evidence, not as formal accuracy claims beyond the included datasets.

## Current Otira normalized summary

Source: `pe3d_current_compare_normalized/summary.json`

- reference rows: `396`
- current `pe3d` rows: `401`
- matched rows: `396`
- `Dmax` MAE: `0.1289 m`
- `Dmed` MAE: `0.0894 m`
- `Dmin` MAE: `0.0781 m`
- `angle_Mview` MAE: `37.62 deg`
- `angle_Xview` MAE: `31.82 deg`

Use the accompanying `README.md`, `summary.json`, and PNG figures inside each subdirectory when citing a specific evaluation snapshot.
