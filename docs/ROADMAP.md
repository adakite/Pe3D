# Pe3d Roadmap

This roadmap defines the next development priorities for `Pe3d` as a standalone Python package for 3D grain and block characterization from point clouds.

## Goal

Keep `Pe3d` as an interpretable, geometry-first workflow that remains scientifically useful while improving robustness, scalability, and packaging quality.

The current package already reproduces MATLAB-level distribution trends reasonably well on the Otira reference case, but per-grain differences remain, especially in preprocessing, segmentation stability, and ellipsoid fitting.

## Guiding Principle

Changes should be evaluated against the current reference bundle before they are accepted.

Reference assets:

- `share_bundle/`
- `evaluation/pe3d_current_compare_normalized/`
- `dist/pe3d-0.1.0.tar.gz`

Success criteria for any major change:

- improve at least one of per-grain agreement, grain counts, or runtime,
- do not degrade the current diameter and angle CDF alignment,
- keep the pipeline explainable and auditable.

## Phase 1: Lock the Current Reference

Keep the current release candidate and evaluation outputs as the baseline for future work.

Main purpose:

- avoid losing the current working state,
- compare future changes against a fixed benchmark,
- preserve a package-ready version while algorithm work continues.

## Phase 2: Fix Preprocessing Parity First

Preprocessing is the first target because small upstream point-selection differences propagate into large per-grain differences.

Priority work:

- make the multiscale local-plane filter closer to the MATLAB behavior,
- review the effective `nscale` logic,
- make denoising closer to MATLAB `pcdenoise`,
- add a controlled `matlab_preprocess` or equivalent test mode.

Target files:

- `src/pe3d/pipeline.py`
- `src/pe3d/math3d.py`
- `src/pe3d/config.py`

## Phase 3: Stabilize Segmentation

Once preprocessing is closer, the next priority is segmentation robustness.

Priority work:

- reduce sensitivity to point density and local noise,
- improve label merging and cleaning,
- keep the current workflow as a `classic` mode,
- add a second segmentation mode based on stronger graph or supervoxel logic.

Target files:

- `src/pe3d/pipeline.py`

## Phase 4: Improve Geometry Only After Segmentation

Ellipsoid fitting should be revisited after segmentation is stable, not before.

Priority work:

- reduce the current minor-axis bias,
- improve axis-ratio agreement,
- improve `Acover`,
- optionally add convex-hull or alpha-shape metrics alongside ellipsoids.

Target files:

- `src/pe3d/pipeline.py`

## Phase 5: Modernize for Large Datasets

Large point clouds are already a practical reason to keep the Python port, since some datasets are not tractable in MATLAB.

Priority work:

- tile-based processing,
- chunked neighbor search,
- optional voxel pyramids,
- parallel grain fitting,
- stronger LAS/LAZ-first workflows.

Target files:

- `src/pe3d/pipeline.py`
- `src/pe3d/io.py`

## Recommendation

Short term:

- complete Phase 2,
- then Phase 3.

Medium term:

- implement Phase 5.

Only after those steps should a learned segmentation path be considered.
