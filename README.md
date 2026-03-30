# pe3d

`pe3d` is a Python reimplementation of the `G3Point` grain-measurement workflow for 3D point clouds.

It is intended for batch processing of patch-scale terrestrial LiDAR or photogrammetric point clouds where the goal is to segment individual grains or blocks and export size, shape, and orientation metrics.

## Status

This repository is the standalone Python project. It is derived from the original `G3Point` methodology, but it does not ship the legacy Python port or the upstream MATLAB source tree.

- Upstream methodological basis: `G3Point`
- This repository's main package: `src/pe3d`
- Validation and benchmarking material: `evaluation/`

## Main capabilities

- load `.ply`, `.las`, and `.laz` point clouds,
- run an ellipsoid-oriented workflow close to the original `G3Point` approach,
- run an alternative block-oriented sizing mode for irregular debris,
- export per-grain tables to CSV and XLSX,
- optionally export per-grain `.ply` files,
- optionally export colored diagnostic point clouds,
- interactively create manual labels and manual block instances for evaluation.

## Installation

```bash
pip install pe3d
```

For local development:

```bash
python -m pip install -e .
```

## Quick start

```bash
pe3d run --pointcloud data/sample_cloud.ply
```

Typical explicit example:

```bash
pe3d run \
  --pointcloud data/sample_cloud.ply \
  --param-csv data/param.csv \
  --output-root pe3d_outputs \
  --sizing-mode ellipsoid \
  --preprocess-mode matlab \
  --workers 1
```

Manual evaluation labeling:

```bash
pe3d label-instances \
  --pointcloud data/sample_cloud.ply \
  --output-root pe3d_labels \
  --reset-session
```

## Outputs

Each run writes MATLAB-style output folders under the selected output root:

- `Figure/<dataset>_nX/`
- `Grain/<dataset>_nX/`
- `Excel/<dataset>_nX/`
- `Cloud/<dataset>_nX/`

## Evaluation

The repository includes comparison artifacts that are useful for a public release because they show what the Python fork currently matches well and where it still differs from archived reference outputs derived from the original workflow.

Recommended starting points:

- `evaluation/pe3d_current_compare_normalized/`
- `evaluation/preprocess_mode_benchmark/`
- `evaluation/worker_benchmark/`
- `evaluation/README.md`

Current Otira normalized comparison highlights:

- matched rows: `396`
- extra Python rows: `5`
- `Dmax` MAE: `0.1289 m`
- `Dmed` MAE: `0.0894 m`
- `Dmin` MAE: `0.0781 m`
- map-view angle MAE: `37.62 deg`
- X-view angle MAE: `31.82 deg`

Interpretation: grain-size parity is substantially closer than orientation parity. Public-facing release notes should say that clearly.

## Repository layout

- `src/pe3d/`: main Python package
- `evaluation/`: validation plots, summaries, and benchmark notes
- `docs/`: internal notes and release guidance

## Attribution and license

`pe3d` is derived from the `G3Point` project and is currently distributed under the inherited repository license terms in [`LICENSE`](LICENSE).

The attribution context and redistribution notes are summarized in [`NOTICE.md`](NOTICE.md).

Relevant upstream reference:

- Steer, P., Guerit, L., Lague, D., Crave, A., and Gourdon, A. (2022). Size, shape and orientation matter: fast and semi-automatic measurement of grain geometries from 3D point clouds. *Earth Surface Dynamics*, 10, 1211-1232. https://doi.org/10.5194/esurf-10-1211-2022
