# G3Point Codebase Documentation

## Purpose and Repository Layout

This repository contains two main code lines for the same grain-analysis workflow:

- `G3point/`: the original MATLAB implementation.
- `src/pe3d/`: the active Python 3.11 package intended for standalone use and PyPI distribution.

Historical parity and evaluation utilities from the earlier Python port remain in `src/g3point/` and `evaluation/`, but they are legacy material and are not part of the maintained `pe3d` package for public release.

The workflow takes a `.ply` point cloud of sediment-covered topography and produces:

- segmented grain labels,
- fitted cuboids and ellipsoids,
- grain-size and orientation statistics,
- exported per-grain `.ply` files,
- summary tables in Excel for MATLAB and CSV/XLSX for Python,
- diagnostic figures.

Reference MATLAB outputs for the bundled Otira dataset live in:

- `G3point/PointCloud/Otira_1cm_grains.ply`
- `G3point/Excel/Otira_1cm_grains_n1/`
- `G3point/Figure/Otira_1cm_grains_n1/`
- `G3point/Grain/Otira_1cm_grains_n1/`

The archived baseline Python port is stored in `archive/python_reference_v1/`.

## MATLAB Code Documentation

### Entry Points and Runtime Model

The MATLAB entry point is `G3point/G3Point.m`. It is written as an interactive script:

1. Adds `Utils/`, `Utils/quadfit/`, and `Utils/geom3d/geom3d/` to the MATLAB path.
2. Opens a file picker unless `param.ptCloudname` is pre-filled.
3. Loads the point cloud with `Utils/loadptCloud.m`.
4. Loads dataset-specific parameters with `defineparameters.m`.
5. Executes the full pipeline and writes outputs beside the MATLAB project.

Typical run:

```matlab
cd G3point
G3Point
```

### Data Loading and Parameter Resolution

`Utils/loadptCloud.m`:

- reads the `.ply` file with `pcread`,
- detects whether RGB colors are present,
- shifts coordinates so the minimum `x`, `y`, and `z` become zero,
- removes invalid points with `removeInvalidPoints`.

`defineparameters.m`:

- reads `PointCloud/param.csv` when a row matches the selected filename,
- otherwise falls back to hard-coded defaults,
- creates new output folders such as `Figure/Otira_1cm_grains_n1/`,
- stores all runtime settings in the `param` structure.

Important parameters are:

- preprocessing toggles: `denoise`, `decimate`, `minima`, `rotdetrend`,
- segmentation: `nnptCloud`, `radfactor`, `maxangle1`,
- cleaning: `maxangle2`, `minflatness`, `minnpoint`,
- fitting: `fitmethod`, `Aquality_thresh`,
- optional sampling: `gridbynumber`, `mindiam`, `naxis`, `dx_gbn`.

### MATLAB Pipeline, Step by Step

#### 1. Preprocessing

`G3Point.m` optionally applies:

- `pcdenoise` for statistical outlier removal,
- `pcdownsample(..., 'gridAverage', param.res)` for voxel-style decimation,
- `disttoplanemultiscale.m` to remove local minima before segmentation,
- planar leveling and quadratic detrending.

`disttoplanemultiscale.m` samples random local neighborhoods across multiple radii, fits local planes with `fitplan.m`, and computes a normalized roughness score. Points below the 95th percentile are retained when minima filtering is enabled.

For detrending, `fitplan.m` estimates a best-fit plane `z = Ax + By + C` using SVD. `adjustnormals3d.m` and `vec2rot.m` rotate the cloud so the mean surface normal points upward, then a quadratic surface is removed from `z`.

#### 2. Segmentation

The segmentation stage is a watershed-like routing on the point cloud graph:

- `knnsearch` builds a `K`-nearest-neighbor graph.
- A per-point local surface proxy is computed from nearest-neighbor spacing.
- `pcnormals` estimates normals and `adjustnormals3d.m` orients them upward.

`segment_labels.m`:

- computes steepest-descent receivers from local slopes,
- identifies sinks where all slopes are uphill,
- recursively collects donors with `addtoStack.m`,
- assigns one label per sink.

#### 3. Merging and Cleaning

`cluster_labels.m` merges over-segmented labels when all three tests pass:

- sink-to-sink distance is small relative to estimated grain radii,
- labels touch in the neighbor graph,
- mean border-normal angle is below `maxangle1`.

The merge itself is performed with `dbscan` on a precomputed label-distance matrix.

`clean_labels.m` performs a second merge using `maxangle2`, then removes:

- labels with fewer than `minnpoint` points,
- labels that are too flat based on singular-value ratios.

#### 4. Geometric Fitting

After segmentation, `G3Point.m` builds a `Pebble` structure containing each grain's points, indices, and surface estimate.

Cuboids are fitted with MATLAB's `pcfitcuboid`.

Ellipsoids are fitted in `Utils/fitellipsoidtograins.m`. Supported methods are:

- `direct`
- `simple`
- `koopmans`
- `inertia`
- `direct_iterative`

The default is `direct`, using the bundled `quadfit` toolbox. The function:

- recenters and rescales each grain,
- fits an implicit ellipsoid,
- converts between implicit and explicit representations,
- computes radii, axes, Euler angles, distance residuals, `r2`, volume, area, and area ratio,
- estimates surface coverage (`Acover`) from random samples on the ellipsoid,
- accepts the fit when `Acover > Aquality_thresh`.

#### 5. Statistics, Orientation, and Export

`grainsizedistribution.m` keeps only fits with `fitok == 1` and `Aqualityok == 1`, then builds:

- diameter distributions for `a`, `b`, `c`,
- volume and area distributions,
- fit-quality metrics such as `Acover`, residual distance, and `r2`.

`ellipsoidorientation3d.m` uses the ellipsoid major axis (`axis1`) to compute:

- `angle_Mview`: map-view azimuth,
- `angle_Xview`: side-view inclination proxy,
- arrow components for rose/compass plots,
- ellipsoid centers.

The sign of the axis is disambiguated using synthetic sensor points far away in `+y` and `+z`.

Exports:

- `exportgrains.m`: one `.ply` per segmented grain,
- `exportgranulotoxls.m`: Excel sheet with center, diameters, and angles,
- `gridbynumbersampling.m`: optional Wolman-style grid-by-number comparison.

### MATLAB Dependencies

The MATLAB code depends on:

- core MATLAB point-cloud support,
- Image Processing Toolbox,
- Lidar Toolbox,
- Parallel Computing Toolbox,
- Statistics and Machine Learning Toolbox,
- bundled third-party packages in `Utils/quadfit/`, `Utils/geom3d/`, and `Utils/minboundbox/`.

One known constraint from the upstream README: the bundled `ellipsoid_distance` must be used, because some external versions do not return the same outputs expected by `fitellipsoidtograins.m`.

## Python Code Documentation

### Package Structure

The active Python package is `Pe3d`, implemented under `src/pe3d/` and installed from `pyproject.toml`.

- `cli.py`: command-line entry point.
- `config.py`: parameter loading and output-path allocation.
- `io.py`: point-cloud reading for PLY/LAS/LAZ, PLY writing, CSV/XLSX export.
- `manual_label.py`: interactive manual labeling GUI and session export.
- `math3d.py`: plane fitting, detrending, normal estimation, vector orientation, sphere sampling.
- `pipeline.py`: full end-to-end workflow.
- `plotting.py`: matplotlib figure generation.

CLI examples:

```bash
.venv/bin/pe3d run --pointcloud path/to/cloud.ply
.venv/bin/pe3d run --pointcloud path/to/cloud.ply --param-csv path/to/param.csv --output-root results
.venv/bin/pe3d run --pointcloud path/to/cloud.ply --sizing-mode block
.venv/bin/pe3d run --pointcloud path/to/cloud.ply --sizing-mode block --ground-mode smrf
.venv/bin/pe3d run --pointcloud path/to/cloud.ply --preprocess-mode matlab
.venv/bin/pe3d run --pointcloud path/to/cloud.ply --save-grains
.venv/bin/pe3d run --pointcloud path/to/cloud.ply --max-ellipsoid-cloud-fraction 0.25
.venv/bin/pe3d run --pointcloud path/to/cloud.ply --workers 1
.venv/bin/pe3d run --pointcloud path/to/cloud.ply --save-colored-clouds
.venv/bin/pe3d run --pointcloud path/to/cloud.ply --sizing-mode block --block-split
.venv/bin/pe3d label --pointcloud path/to/cloud.ply --output-root manual_labels
.venv/bin/pe3d label-instances --pointcloud path/to/cloud.ply --output-root manual_labels
```

### Configuration and I/O

`config.py` mirrors the MATLAB parameter model:

- default values match `defineparameters.m`,
- `load_parameters()` reads `param.csv`,
- the MATLAB `cf` column is mapped to Python `radfactor`,
- `minnpoint` is computed as `max(nnptcloud, nmin)`,
- `create_run_paths()` allocates `Figure/`, `Grain/`, and `Excel/` directories under a configurable output root.
- `create_run_paths()` also allocates `Cloud/` for colorized point-cloud exports.

`io.py`:

- reads `.ply` files with `plyfile` and `.las`/`.laz` files with `laspy`,
- preserves extra vertex fields,
- detects RGB fields,
- shifts coordinates to a zero minimum, as MATLAB does,
- writes per-grain `.ply`,
- writes colorized `.ply` exports of the original point cloud,
- exports both CSV and XLSX granulometry tables.

`manual_label.py`:

- loads the original point cloud without running the segmentation pipeline,
- fits the same dominant plane used for horizontal normalization,
- optionally applies the quadratic detrend,
- opens interactive map-view polygon labeling sessions,
- supports both zone labels and per-block instance labels,
- saves point-wise labels, polygon definitions, metadata, and a preview `.ply`.

### Python Pipeline

The end-to-end workflow is implemented in `pipeline.py` through `run_pipeline()`.

Two sizing backends are available:

- `ellipsoid`: the original G3Point-like mode for rounded grains.
- `block`: a second mode for irregular protruding blocks or debris.

#### 1. Preprocessing

- `statistical_denoise()` replaces MATLAB `pcdenoise`.
- `voxel_downsample()` replaces `pcdownsample`.
- `multiscale_plan_distance()` supports two modes:
  `matlab` for a closer match to the original G3Point denoise and minima filter, and `fast` for the earlier lightweight approximation.
- oversize ellipsoids can be rejected with `--max-ellipsoid-cloud-fraction`, measured against the processed cloud maximum span.
- per-grain fitting supports worker threads through `--workers`; `1` keeps serial fitting and `0` enables auto-threaded fitting.
- `--save-colored-clouds` exports three RGB point clouds colored by fitted ellipsoid volume, elongation, and azimuth.
- `--ground-mode smrf` adds a simple morphological ground/background filter before block segmentation and can export `height_above_ground` and `ground_mask` clouds.
- `--block-split` enables an experimental local map-view watershed inside oversized merged `block` components; it is disabled by default because it was tuned on `Nuage_HP` manual instances and is not yet a global default.
- `fit_plane()` and `detrend_quadratic()` reproduce the detrending stage.

#### 2. Segmentation and Cleaning

The segmentation logic follows the MATLAB design:

- `segment_labels()` performs steepest-descent routing on the rotated cloud.
- `cluster_labels()` merges labels using sink distance, adjacency, and border-normal angles.
- `clean_labels()` performs a second angle-based merge, then removes small and too-flat labels.

The Python implementation uses vectorized NumPy operations, `cKDTree`, and graph connected components instead of MATLAB cell arrays plus `dbscan`.

In `block` mode, `segment_blocks()` replaces the watershed-style route-and-merge stage. It:

- estimates local top-hat elevation, local relief, local vertical spread, local scattering, and multiscale PCA curvature / planarity from point neighborhoods,
- combines those features into a blockiness score,
- optionally prefilters the cloud with a SMRF-like ground/background model in the horizontal frame,
- keeps high-score points as block support points,
- extracts connected components with a radius graph,
- grows components back onto nearby moderate-score points,
- rejects components that are too small or too flat.

With `--block-split`, the largest merged components are then split in map view with a local watershed seeded from the strongest blockiness peaks.

### Manual Labeling Workflow

The `pe3d label` subcommand is intended to build human-reviewed reference masks for datasets where automated sizing remains uncertain.

Workflow:

1. Load the original `.ply`, `.las`, or `.laz`.
2. Fit a dominant plane and rotate the cloud into a horizontal frame.
3. Display:
   - the original 3D cloud,
   - a rotated map view.
4. Use polygon selection in map view to define:
   - one `block` polygon,
   - one `non-block` polygon.
   Drawing a new polygon for a mode replaces the previous one.
   `--reset-session` clears any previously saved polygons before the GUI opens.
5. Save the session as:
   - `*_manual_labels.npz`,
   - `*_manual_labels.json`,
   - `*_manual_polygons.json`,
   - `*_manual_labels.ply`.

The stored labels are point-wise and use:

- `1` for block,
- `-1` for non-block,
- `0` for unlabeled / unknown.

### Manual Instance Workflow

The `pe3d label-instances` subcommand is intended for manually isolating individual blocks when zone labels are not sufficient.

Workflow:

1. Load the original `.ply`, `.las`, or `.laz`.
2. Fit a dominant plane and rotate the cloud into a horizontal frame.
3. Display:
   - the original 3D cloud,
   - a rotated map view.
4. Draw one polygon per block.
   Each completed polygon creates one new positive integer instance id.
5. Save the session as:
   - `*_manual_instances.npz`,
   - `*_manual_instances.json`,
   - `*_manual_instances_polygons.json`,
   - `*_manual_instances.ply`.

The stored instance labels use:

- `0` for unlabeled / unknown,
- `1, 2, 3, ...` for individual block ids.

#### 3. Geometric Fitting

`fit_pca_cuboid()` provides a PCA-based oriented bounding box approximation to MATLAB `pcfitcuboid`.

The cuboid and ellipsoid fitting stage now supports threaded execution, but the current benchmark in `evaluation/worker_benchmark/` showed that serial fitting remained faster than auto-threaded fitting on both Otira and `Nuage_HP` while producing identical CSVs. For that reason, the CLI default stays at `--workers 1`.

In `block` mode, `_fit_block_model()` uses the cuboid axes as `Dmax/Dmed/Dmin` and computes volume/area from a convex hull when possible, falling back to the oriented bounding box if hull construction is unstable.

Ellipsoid fitting is implemented in `_fit_ellipsoid()` with three strategies:

- `pca_maxabs`: PCA axes with radii from the maximum absolute local coordinates,
- `bounded_pca`: PCA initialization followed by bounded nonlinear least-squares refinement,
- `hybrid_direct`: bounded PCA plus a direct algebraic ellipsoid fit, with acceptance gating.

The current CLI default is `hybrid_direct`.

Fit finalization computes the same main outputs as MATLAB:

- center,
- sorted radii,
- rotation matrix and axis vectors,
- residual distance,
- `r2`,
- volume and area,
- `Aratio`,
- `Acover`.

The Python `Acover` estimate uses a deterministic Fibonacci-sphere sampling rather than random ellipsoid samples.

#### 4. Statistics, Orientation, and Export

`grain_size_distribution()` mirrors the MATLAB granulometry tables for accepted ellipsoids.

`ellipsoid_orientation()` computes map-view and x-view angles. It preserves the MATLAB sign-disambiguation idea, but supports two angle formulas:

- `matlab`
- `atan2`

The current default is `atan2`.

`plotting.py` regenerates the main figure families:

- elevation,
- initial/clustered/cleaned labels,
- fitted cuboids,
- fitted ellipsoids,
- size, area, axis-ratio, cover, and orientation plots.

### Python-Specific Additions

The `Pe3d` package includes features that do not exist in the original MATLAB entry point:

- a repeatable CLI,
- installable package metadata in `pyproject.toml`,
- configurable output roots,
- JSON-friendly summaries from the CLI.

Repository-level parity artifacts still exist outside the package:

- `archive/python_reference_v1/`
- `evaluation/`
- `src/g3point/`

One important gap remains: the MATLAB `gridbynumbersampling.m` stage is not yet ported.

## What Differs Between MATLAB and Python

### Method-Level Differences

| Area | MATLAB | Python | Practical Effect |
| --- | --- | --- | --- |
| Execution model | Interactive script with file picker | CLI package with subcommands | Python is easier to batch, script, and validate |
| Output location | Always inside `G3point/` | Configurable `--output-root` | Safer for repeated experiments |
| Multiscale minima removal | `rangesearch` + `parfor` over random circles | `cKDTree` + sampled neighborhoods | Same intent, different neighborhood sampling details |
| Label merging | `dbscan` on a precomputed merge matrix | connected components on a boolean adjacency graph | Similar grouping behavior, simpler runtime dependency in Python |
| Merge threshold | raw `radfactor` from CSV | `radfactor` with `RADFACTOR_CORRECTION = 5/6` | Calibrated to keep Otira segmentation closer to MATLAB |
| Cuboid fitting | `pcfitcuboid` | PCA-oriented bounding box | Python cuboids are approximate, not toolbox-identical |
| Ellipsoid fitting | quadfit toolbox, default `direct` | native bounded PCA + direct algebraic hybrid | Python is reimplemented, not a line-by-line port |
| Surface cover | random ellipsoid samples | deterministic Fibonacci-sphere samples | Less run-to-run noise in Python |
| Orientation formula | `atan(v/u)+pi/2` and `atan(v/w)+pi/2` | selectable `matlab` or `atan2` modes | Python can be more numerically stable |
| Grid-by-number | implemented | not yet implemented | MATLAB still has one analysis stage missing in Python |
| Validation | manual inspection | external repository tooling, not part of `pe3d` | package stays cleaner for standalone distribution |

### Language- and Runtime-Specific Differences

#### Data Model

MATLAB uses:

- `pointCloud` objects,
- structs such as `param`, `Pebble`, `Ellipsoidm`, `granulo`,
- cell arrays for label stacks.

Python uses:

- `numpy.ndarray` for coordinates and normals,
- dataclasses (`Parameters`, `RunPaths`, `PipelineResult`),
- dictionaries for fitted models and granulometry,
- `Path` objects for filesystem operations.

#### Indexing and Missing Labels

MATLAB is 1-based and often uses `NaN` for discarded labels. Python is 0-based internally, but exported grain numbering still starts at 1. Discarded labels are represented with `0` in label arrays rather than `NaN`.

#### Libraries

MATLAB depends on toolbox functions such as `pcnormals`, `knnsearch`, `pcfitcuboid`, `dbscan`, `xlswrite`, and `pcwrite`.

Python replaces them with:

- `scipy.spatial.cKDTree`,
- `numpy.linalg`,
- `scipy.optimize.least_squares`,
- `pandas`,
- `plyfile`,
- `matplotlib`.

#### Packaging and Reproducibility

MATLAB relies on `addpath` and the current working directory. Python is packaged with `pyproject.toml`, versioned dependencies, and a `pe3d` console script, which makes automation and CI-style execution easier.

#### Performance Model

MATLAB uses optimized toolbox functions and `parfor` in the multiscale plane-distance routine. Python is currently tuned for correctness first; performance-critical sections are vectorized where practical, but not yet optimized to match MATLAB runtime in all cases.

### Current Parity Status

For the bundled Otira dataset, the Python port is end-to-end functional and close in grain count and size metrics, but orientation remains the largest difference.

The current matched evaluation is in `evaluation/orientation_focus_compare/summary.json`. Against the MATLAB reference:

- MATLAB accepted ellipsoids: 396
- current Python accepted ellipsoids: 403
- center-distance mean: 0.162 m
- `Dmax` MAE: 0.118 m
- `Dmed` MAE: 0.080 m
- `Dmin` MAE: 0.071 m
- `angle_Mview` MAE: 37.83 degrees
- `angle_Xview` MAE: 31.43 degrees

That means the size fit is reasonably close, while orientation still needs additional work.

## Where to Modify the Code

For future development, the main edit points are:

- MATLAB segmentation: `Utils/segment_labels.m`, `Utils/cluster_labels.m`, `Utils/clean_labels.m`
- MATLAB ellipsoid fitting: `Utils/fitellipsoidtograins.m`
- MATLAB orientation: `Utils/ellipsoidorientation3d.m`
- Python segmentation and fitting: `src/pe3d/pipeline.py`
- Python geometry helpers: `src/pe3d/math3d.py`
- Python I/O and exports: `src/pe3d/io.py`
- legacy parity tooling: `src/g3point/validation.py`, `src/g3point/compare.py`

This split is intentional: the MATLAB tree remains the original reference implementation, `pe3d` is the standalone Python package, and the older parity utilities remain outside the packaged surface.
