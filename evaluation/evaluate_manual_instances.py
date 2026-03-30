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


def _prepare_points(
    pointcloud_path: Path,
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
    rng = np.random.default_rng(0)

    raw_points = cloud.points.copy()
    points = raw_points.copy()
    if params.denoise:
        if preprocess_mode == "matlab":
            denoise_mask = statistical_denoise(points, MATLAB_DENOISE_NEIGHBOURS, MATLAB_DENOISE_STD_THRESHOLD)
        elif preprocess_mode == "fast":
            denoise_mask = statistical_denoise(points, params.nnptcloud, 2.0)
        else:
            raise ValueError(f"Unsupported preprocess mode: {preprocess_mode}")
        points = points[denoise_mask]
    if params.decimate:
        points = voxel_downsample(points, params.res)
    if params.minima:
        roughness = multiscale_plan_distance(points, params, rng, preprocess_mode)
        points = points[roughness < np.percentile(roughness, 95.0)]

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

    labels, _, diagnostics = segment_blocks(
        points,
        points_rot,
        params,
        neighbour_idx,
        distances,
        enable_split=block_split,
        prefilter_mask=None if smrf_result is None else np.asarray(smrf_result["nonground_mask"], dtype=bool),
    )
    _, raw_nearest = tree.query(raw_points, k=1, workers=-1)
    raw_labels = labels[np.asarray(raw_nearest, dtype=np.int64)]
    return raw_labels, raw_points, diagnostics


def _summarize_instances(manual_labels: np.ndarray, auto_labels: np.ndarray, diagnostics: dict[str, float]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for instance_id in np.unique(manual_labels):
        if instance_id <= 0:
            continue
        manual_mask = manual_labels == instance_id
        overlapping_labels, counts = np.unique(auto_labels[manual_mask], return_counts=True)
        order = np.argsort(counts)[::-1]
        overlapping_labels = overlapping_labels[order]
        counts = counts[order]

        top_auto = [[int(label), int(count)] for label, count in zip(overlapping_labels[:5], counts[:5], strict=False)]

        best_label = 0
        best_overlap = 0
        best_precision = 0.0
        best_recall = 0.0
        best_iou = 0.0

        for label, overlap in zip(overlapping_labels, counts, strict=False):
            if label == 0:
                continue
            auto_mask = auto_labels == label
            union = np.count_nonzero(manual_mask | auto_mask)
            precision = float(overlap / np.count_nonzero(auto_mask)) if np.count_nonzero(auto_mask) else 0.0
            recall = float(overlap / np.count_nonzero(manual_mask)) if np.count_nonzero(manual_mask) else 0.0
            iou = float(overlap / union) if union else 0.0
            if iou > best_iou:
                best_label = int(label)
                best_overlap = int(overlap)
                best_precision = precision
                best_recall = recall
                best_iou = iou

        rows.append(
            {
                "instance_id": int(instance_id),
                "manual_points": int(np.count_nonzero(manual_mask)),
                "background_overlap": int(counts[overlapping_labels == 0][0]) if np.any(overlapping_labels == 0) else 0,
                "best_nonzero_label": best_label,
                "best_nonzero_overlap_points": best_overlap,
                "best_nonzero_recall": best_recall,
                "best_nonzero_precision": best_precision,
                "best_nonzero_iou": best_iou,
                "top_auto_overlaps": top_auto,
            }
        )

    nonzero_ious = [float(row["best_nonzero_iou"]) for row in rows]
    nonzero_recalls = [float(row["best_nonzero_recall"]) for row in rows]
    return {
        "diagnostics": {key: float(value) for key, value in diagnostics.items()},
        "rows": rows,
        "mean_best_nonzero_iou": float(np.mean(nonzero_ious)) if nonzero_ious else 0.0,
        "mean_best_nonzero_recall": float(np.mean(nonzero_recalls)) if nonzero_recalls else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pe3d block segmentation against manual instance labels.")
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument("--manual-instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--param-csv", type=Path, default=None)
    parser.add_argument("--preprocess-mode", choices=["matlab", "fast"], default="matlab")
    parser.add_argument("--block-split", action="store_true", help="Enable the experimental large-component split stage.")
    parser.add_argument("--ground-mode", choices=["none", "smrf"], default="none")
    parser.add_argument("--smrf-cell-size", type=float, default=0.0)
    parser.add_argument("--smrf-max-window", type=float, default=12.0)
    parser.add_argument("--smrf-height-threshold", type=float, default=0.35)
    parser.add_argument("--smrf-slope-threshold", type=float, default=0.75)
    args = parser.parse_args()

    manual_labels = np.load(args.manual_instances)["labels"]
    auto_labels, _, diagnostics = _prepare_points(
        args.pointcloud,
        args.param_csv,
        args.preprocess_mode,
        block_split=args.block_split,
        ground_mode=args.ground_mode,
        smrf_cell_size=args.smrf_cell_size,
        smrf_max_window=args.smrf_max_window,
        smrf_height_threshold=args.smrf_height_threshold,
        smrf_slope_threshold=args.smrf_slope_threshold,
    )
    summary = _summarize_instances(manual_labels, auto_labels, diagnostics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mean_best_nonzero_iou": summary["mean_best_nonzero_iou"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
