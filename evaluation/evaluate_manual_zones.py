from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pe3d.config import load_parameters
from pe3d.io import load_pointcloud
from pe3d.math3d import detrend_quadratic, fit_plane, normalize, orient_vectors, rotation_matrix_between
from pe3d.pipeline import (
    MATLAB_DENOISE_NEIGHBOURS,
    MATLAB_DENOISE_STD_THRESHOLD,
    multiscale_plan_distance,
    segment_blocks,
    smrf_ground_model,
    statistical_denoise,
    voxel_downsample,
)


def _prepare_labels(
    pointcloud_path: Path,
    manual_labels_path: Path,
    param_csv: Path | None,
    preprocess_mode: str,
    *,
    block_split: bool,
    ground_mode: str,
    smrf_cell_size: float,
    smrf_max_window: float,
    smrf_height_threshold: float,
    smrf_slope_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    params = load_parameters(pointcloud_path, param_csv=param_csv)
    cloud = load_pointcloud(pointcloud_path)
    raw_points = cloud.points.copy()
    raw_labels = np.load(manual_labels_path)["labels"]
    if raw_labels.shape[0] != raw_points.shape[0]:
        raise ValueError("Manual labels size does not match point cloud size.")

    rng = np.random.default_rng(0)
    points = raw_points.copy()
    labels = raw_labels.copy()
    if params.denoise:
        if preprocess_mode == "matlab":
            denoise_mask = statistical_denoise(points, MATLAB_DENOISE_NEIGHBOURS, MATLAB_DENOISE_STD_THRESHOLD)
        elif preprocess_mode == "fast":
            denoise_mask = statistical_denoise(points, params.nnptcloud, 2.0)
        else:
            raise ValueError(f"Unsupported preprocess mode: {preprocess_mode}")
        points = points[denoise_mask]
        labels = labels[denoise_mask]
    if params.decimate:
        points = voxel_downsample(points, params.res)
        tree = cKDTree(raw_points)
        _, nearest_idx = tree.query(points, k=1, workers=-1)
        labels = raw_labels[np.asarray(nearest_idx, dtype=np.int64)]
    if params.minima:
        roughness = multiscale_plan_distance(points, params, rng, preprocess_mode)
        minima_mask = roughness < np.percentile(roughness, 95.0)
        points = points[minima_mask]
        labels = labels[minima_mask]

    points_rot = points.copy()
    if params.rotdetrend:
        a, b, _, _, _ = fit_plane(points)
        normal = normalize(np.array([[-a, -b, 1.0]], dtype=np.float64))[0]
        normal = orient_vectors(np.zeros((1, 3)), normal.reshape(1, 3), np.array([0.0, 0.0, 1e32]))[0]
        rotation = rotation_matrix_between(normal, np.array([0.0, 0.0, 1.0]))
        centre = points.mean(axis=0)
        points_rot = (points - centre) @ rotation.T + centre
        points_rot, _ = detrend_quadratic(points_rot)

    tree = cKDTree(points)
    distances, neighbour_idx = tree.query(points, k=min(params.nnptcloud + 1, points.shape[0]), workers=-1)
    if neighbour_idx.ndim == 1:
        neighbour_idx = neighbour_idx[:, None]
        distances = distances[:, None]
    if neighbour_idx.shape[1] > 1:
        neighbour_idx = neighbour_idx[:, 1:]
        distances = distances[:, 1:]

    smrf_result = None
    if ground_mode == "smrf":
        smrf_result = smrf_ground_model(
            points_rot,
            distances,
            cell_size=smrf_cell_size,
            max_window=smrf_max_window,
            height_threshold=smrf_height_threshold,
            slope_threshold=smrf_slope_threshold,
        )
    elif ground_mode != "none":
        raise ValueError(f"Unsupported ground mode: {ground_mode}")

    block_labels, _, diagnostics = segment_blocks(
        points,
        points_rot,
        params,
        neighbour_idx,
        distances,
        enable_split=block_split,
        prefilter_mask=None if smrf_result is None else np.asarray(smrf_result["nonground_mask"], dtype=bool),
    )
    predicted_positive = block_labels > 0
    labelled_mask = labels != 0
    return labels, predicted_positive, diagnostics


def _summarize(labels: np.ndarray, predicted_positive: np.ndarray, diagnostics: dict[str, float]) -> dict[str, object]:
    labelled_mask = labels != 0
    block_mask = labels == 1
    non_block_mask = labels == -1

    tp = int(np.count_nonzero(predicted_positive & block_mask))
    fp = int(np.count_nonzero(predicted_positive & non_block_mask))
    fn = int(np.count_nonzero(~predicted_positive & block_mask))
    tn = int(np.count_nonzero(~predicted_positive & non_block_mask))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "manual_block_points": int(np.count_nonzero(block_mask)),
        "manual_non_block_points": int(np.count_nonzero(non_block_mask)),
        "predicted_positive_points_in_labeled_zones": int(np.count_nonzero(predicted_positive & labelled_mask)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "diagnostics": {key: float(value) for key, value in diagnostics.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pe3d block segmentation against manual block/non-block zones.")
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument("--manual-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--param-csv", type=Path, default=None)
    parser.add_argument("--preprocess-mode", choices=["matlab", "fast"], default="matlab")
    parser.add_argument("--block-split", action="store_true", help="Enable the experimental block split stage.")
    parser.add_argument("--ground-mode", choices=["none", "smrf"], default="none")
    parser.add_argument("--smrf-cell-size", type=float, default=0.0)
    parser.add_argument("--smrf-max-window", type=float, default=12.0)
    parser.add_argument("--smrf-height-threshold", type=float, default=0.35)
    parser.add_argument("--smrf-slope-threshold", type=float, default=0.75)
    args = parser.parse_args()

    labels, predicted_positive, diagnostics = _prepare_labels(
        args.pointcloud,
        args.manual_labels,
        args.param_csv,
        args.preprocess_mode,
        block_split=args.block_split,
        ground_mode=args.ground_mode,
        smrf_cell_size=args.smrf_cell_size,
        smrf_max_window=args.smrf_max_window,
        smrf_height_threshold=args.smrf_height_threshold,
        smrf_slope_threshold=args.smrf_slope_threshold,
    )
    summary = _summarize(labels, predicted_positive, diagnostics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "balanced_accuracy": summary["balanced_accuracy"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
