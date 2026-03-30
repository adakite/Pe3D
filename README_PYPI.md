# Pe3d

Pe3d is a Python 3.11 command-line tool for extracting pebble or grain metrics from 3D point clouds.

It is a Python fork and reimplementation of the `G3Point` MATLAB workflow, packaged as a standalone CLI.

It provides an end-to-end workflow for:

- preprocessing `.ply`, `.las`, and `.laz` point clouds,
- watershed-style grain segmentation,
- grain merging and cleaning,
- cuboid and ellipsoid fitting,
- grain-size and orientation statistics,
- export of per-grain `.ply` files and summary tables,
- interactive manual block / non-block labeling for evaluation datasets.

## Installation

```bash
pip install pe3d
```

## Quick Start

```bash
pe3d run --pointcloud path/to/cloud.ply
```

Optional arguments:

- `--param-csv`: dataset parameter table matching the MATLAB-style `param.csv` format
- `--output-root`: output directory root, default `pe3d_outputs`
- `--sizing-mode`: `ellipsoid` for rounded grains or `block` for irregular protruding blocks
- `--fit-strategy`: `pca_maxabs`, `bounded_pca`, or `hybrid_direct`
- `--orientation-mode`: `atan2` or `matlab`
- `--preprocess-mode`: `matlab` or `fast`
- `--ground-mode`: `none` or `smrf` for an optional morphological ground/background filter in block mode
- `--smrf-cell-size`: SMRF grid cell size, use `0` for automatic sizing from point spacing
- `--smrf-max-window`: maximum SMRF opening window in cloud units
- `--smrf-height-threshold`: base height threshold above the SMRF surface
- `--smrf-slope-threshold`: slope-dependent additive threshold multiplier for SMRF
- `--max-ellipsoid-cloud-fraction`: reject ellipsoids larger than this fraction of the processed cloud max span, default `0.25`, use `0` to disable
- `--workers`: per-grain fitting worker threads, default `1`, use `0` for auto
- `--save-grains` or `--no-save-grains`: export per-grain `.ply` files, default off
- `--save-colored-clouds` or `--no-save-colored-clouds`: export the original point cloud recolored by ellipsoid volume, elongation, and azimuth, default off
- `--block-split` or `--no-block-split`: experimental local watershed splitter for oversized merged block components, default off
- `--skip-plots`

Example:

```bash
pe3d run \
  --pointcloud data/sample_cloud.laz \
  --param-csv data/param.csv \
  --sizing-mode block \
  --preprocess-mode matlab \
  --ground-mode smrf \
  --max-ellipsoid-cloud-fraction 0.25 \
  --no-block-split \
  --workers 1 \
  --save-colored-clouds \
  --save-grains \
  --output-root results
```

`ellipsoid` mode is the original G3Point-style workflow and is best suited to rounded pebbles or grains.

`block` mode is an alternative backend for irregular debris or avalanche deposits. It combines local relief, local vertical spread, local scattering, and multiscale PCA curvature / planarity into a blockiness score, segments connected components from that score, then measures each object with oriented bounding-box axes and convex-hull volume/area instead of relying on an ellipsoid fit. `--ground-mode smrf` adds a simple morphological ground/background filter before segmentation. The optional `--block-split` stage adds a local map-view watershed inside only the largest merged components. It remains disabled by default because it was tuned on `Nuage_HP` manual instances and has not yet been validated on other datasets.

## Manual Labeling

Use `pe3d label` to open an interactive labeling window for manual block annotation:

```bash
pe3d label \
  --pointcloud data/sample_cloud.ply \
  --output-root manual_labels \
  --reset-session \
  --max-display-points 120000 \
  --rotate-horizontal
```

The tool fits a dominant plane, rotates the cloud into a horizontal map view, and opens:

- a 3D inspection pane,
- a top-down map view where polygon selection happens.

The simplified session model stores one polygon for `block` and one polygon for `non-block`. Drawing a new polygon for a mode replaces the previous one. `Clear Mode` removes the currently selected zone.

Use `--reset-session` to ignore and delete any previously saved polygons for that dataset.

The session saves:

- `*_manual_labels.npz`: point-wise label array,
- `*_manual_labels.json`: metadata and counts,
- `*_manual_polygons.json`: stored block / non-block polygons,
- `*_manual_labels.ply`: RGB preview cloud for CloudCompare or similar viewers.

Use `pe3d label-instances` when you want to isolate individual blocks instead of broad zones:

```bash
pe3d label-instances \
  --pointcloud data/sample_cloud.ply \
  --output-root manual_labels \
  --reset-session \
  --max-display-points 120000 \
  --rotate-horizontal
```

In this mode, each polygon creates one block instance id. The session saves:

- `*_manual_instances.npz`: point-wise integer instance labels,
- `*_manual_instances.json`: metadata and counts,
- `*_manual_instances_polygons.json`: one polygon per saved instance,
- `*_manual_instances.ply`: RGB preview cloud with one color per instance.

## Performance Note

`Pe3d` uses multithreaded `cKDTree` queries internally, and per-grain fitting can also be threaded with `--workers`.

In the current benchmark on this repository, serial fitting (`--workers 1`) was faster than auto-threaded fitting (`--workers 0`) on both Otira and `Nuage_HP`, while producing identical CSV outputs. See:

- `evaluation/worker_benchmark/README.md`

## Validation Snapshot

This repository includes evaluation material comparing the current Python implementation to bundled MATLAB reference outputs.

On the normalized Otira comparison snapshot included in `evaluation/pe3d_current_compare_normalized/summary.json`, the current Python release candidate matches `396` reference rows with:

- `Dmax` MAE: `0.1289 m`
- `Dmed` MAE: `0.0894 m`
- `Dmin` MAE: `0.0781 m`
- `angle_Mview` MAE: `37.62 deg`
- `angle_Xview` MAE: `31.82 deg`

That snapshot indicates stronger size parity than orientation parity, which is the main remaining gap relative to the MATLAB workflow.

## Inputs and Outputs

Input point clouds can be `.ply`, `.las`, or `.laz` and must provide `x`, `y`, and `z` coordinates. Optional RGB fields are preserved on load when present. Per-grain exports remain `.ply`.

Each run creates:

- `Figure/<dataset>_nX/`
- `Grain/<dataset>_nX/`
- `Excel/<dataset>_nX/`
- `Cloud/<dataset>_nX/`

The summary table is exported as both CSV and XLSX.

When `--save-colored-clouds` is enabled, `Cloud/<dataset>_nX/` contains three RGB `.ply` files built from the original point set:

- `*_by_volume.ply`
- `*_by_elongation.ply`
- `*_by_azimuth.ply`

When `--ground-mode smrf` is active, the same folder also contains:

- `*_height_above_ground.ply`
- `*_ground_mask.ply`

## License and Attribution

`Pe3d` is currently distributed under the same license terms as the original `G3Point` MATLAB code from which this Python fork is derived.

That means redistribution and modification are permitted, but the original copyright notice, conditions, and disclaimer must be retained. The original license also requires scientific or technical publications that benefit from the software to acknowledge its use and cite the corresponding publication.

See:

- `LICENSE`
- `NOTICE.md`

Attribution and citation source:

- G3Point software page: https://lidar.univ-rennes.fr/en/g3point
- Steer, Guérit, Lague, Crave, and Gourdon (2022), https://doi.org/10.5194/esurf-10-1211-2022
