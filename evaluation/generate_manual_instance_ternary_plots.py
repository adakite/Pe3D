from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.tri as tri
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

from pe3d.io import load_pointcloud


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _calculate_barycentric_coordinates(eigenvalues: np.ndarray) -> np.ndarray:
    delta = np.sqrt(2.0) / 2.0 * (eigenvalues[:, 0] - eigenvalues[:, 1])
    delta_max = np.sqrt(2.0) / 2.0
    c_coord = eigenvalues[:, 2] / (1.0 / 3.0)
    a_coord = (1.0 - c_coord) * delta / delta_max
    b_coord = 1.0 - (a_coord + c_coord)
    x_coord = 0.5 * (2.0 * b_coord + c_coord) / (a_coord + b_coord + c_coord)
    y_coord = np.sqrt(3.0) / 2.0 * c_coord / (a_coord + b_coord + c_coord)
    return np.vstack((x_coord, y_coord))


def _plot_ternary(x_coord: np.ndarray, y_coord: np.ndarray, ax: plt.Axes) -> None:
    points = np.vstack((x_coord, y_coord))
    points = points[:, ~np.any(np.isnan(points), axis=0)]
    if points.shape[1] == 0:
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x_coord, y_coord = points
    try:
        density = gaussian_kde(points)(points)
    except Exception:
        density = np.ones_like(x_coord)

    mappable = ax.scatter(x_coord, y_coord, c=density, s=2.0, rasterized=True)

    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) * 0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=3)
    ax.triplot(trimesh, color="0.6", linestyle="-", linewidth=0.5, zorder=0)
    ax.plot(
        np.hstack((corners[:, 0], corners[0, 0])),
        np.hstack((corners[:, 1], corners[0, 1])),
        color="k",
        linestyle="-",
        linewidth=plt.rcParams["axes.linewidth"],
        zorder=0,
    )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-0.015, -0.015, "1D", ha="right", va="top")
    ax.text(1.015, -0.015, "2D", ha="left", va="top")
    ax.text(0.5, np.sqrt(3.0) * 0.5 + 0.015, "3D", ha="center", va="bottom")
    plt.colorbar(mappable, ax=ax, orientation="horizontal", shrink=0.8, pad=0.04)


def _block_bbox_diagonal(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    return float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))


def _min_positive_nearest_distance(points: np.ndarray) -> float:
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2, workers=-1)
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if nearest.size == 0:
        raise RuntimeError("Unable to determine a positive nearest-neighbour distance.")
    return float(np.min(nearest))


def _batched_eigenvalues(
    points: np.ndarray,
    diameter: float,
    *,
    n_min_points: int = 10,
    batch_size: int = 256,
) -> np.ndarray:
    tree = cKDTree(points)
    radius = float(diameter) * 0.5
    eigenvalues = np.full((points.shape[0], 3), np.nan, dtype=np.float64)
    for start in range(0, points.shape[0], batch_size):
        stop = min(start + batch_size, points.shape[0])
        neighbours = tree.query_ball_point(points[start:stop], radius, workers=-1)
        for local_idx, indices in enumerate(neighbours):
            if len(indices) < n_min_points:
                continue
            neighbourhood = points[np.asarray(indices, dtype=np.int64)]
            covariance = np.cov(neighbourhood, rowvar=False)
            values = np.linalg.eigvalsh(covariance)[::-1]
            total = float(np.sum(values))
            if total <= 0.0:
                continue
            eigenvalues[start + local_idx] = values / total
    return eigenvalues


def _format_scale(value: float) -> str:
    return f"{value:.3f} m" if value < 10.0 else f"{value:.2f} m"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ternary PCA plots for each manually labeled block.")
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument("--manual-instances", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-count", type=int, default=4)
    parser.add_argument("--n-min-points", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pointcloud = load_pointcloud(args.pointcloud)
    points = pointcloud.points
    manual = np.load(args.manual_instances)
    labels = np.asarray(manual["labels"], dtype=np.int32)
    block_ids = [int(block_id) for block_id in np.unique(labels) if block_id > 0]
    if not block_ids:
        raise RuntimeError("No positive block instance labels found.")

    min_positive_spacing = _min_positive_nearest_distance(points)
    lower_scale = 3.0 * min_positive_spacing
    block_sizes = {block_id: _block_bbox_diagonal(points[labels == block_id]) for block_id in block_ids}
    upper_scale = max(block_sizes.values())
    if not np.isfinite(lower_scale) or not np.isfinite(upper_scale) or upper_scale <= lower_scale:
        raise RuntimeError("Invalid scale bounds for ternary computation.")
    scales = np.geomspace(lower_scale, upper_scale, args.scale_count)

    metadata: dict[str, object] = {
        "pointcloud": str(args.pointcloud),
        "manual_instances": str(args.manual_instances),
        "point_count": int(points.shape[0]),
        "block_ids": block_ids,
        "block_point_counts": {str(block_id): int(np.count_nonzero(labels == block_id)) for block_id in block_ids},
        "block_bbox_diagonals_m": {str(block_id): float(block_sizes[block_id]) for block_id in block_ids},
        "min_positive_nearest_distance_m": float(min_positive_spacing),
        "lower_scale_m": float(lower_scale),
        "upper_scale_m": float(upper_scale),
        "scales_m": [float(scale) for scale in scales],
        "n_min_points": int(args.n_min_points),
        "batch_size": int(args.batch_size),
        "plots": {},
    }

    rows = 2
    cols = int(np.ceil(args.scale_count / rows))
    for block_id in block_ids:
        block_points = points[labels == block_id]
        fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.5 * rows), constrained_layout=True)
        axes_array = np.atleast_1d(axes).ravel()
        block_meta: list[dict[str, object]] = []
        for scale_idx, scale in enumerate(scales):
            eigenvalues = _batched_eigenvalues(
                block_points,
                scale,
                n_min_points=args.n_min_points,
                batch_size=args.batch_size,
            )
            valid_mask = np.isfinite(eigenvalues).all(axis=1)
            valid_count = int(np.count_nonzero(valid_mask))
            ternary = _calculate_barycentric_coordinates(eigenvalues[valid_mask]) if valid_count else np.empty((2, 0))
            axis = axes_array[scale_idx]
            if valid_count:
                _plot_ternary(ternary[0], ternary[1], ax=axis)
            else:
                axis.text(0.5, 0.5, "No valid neighbourhoods", ha="center", va="center", transform=axis.transAxes)
                axis.set_axis_off()
            axis.set_title(f"Scale {scale_idx + 1}: {_format_scale(float(scale))}\nvalid {valid_count}/{block_points.shape[0]}")
            block_meta.append(
                {
                    "scale_index": int(scale_idx + 1),
                    "diameter_m": float(scale),
                    "valid_point_count": valid_count,
                    "total_point_count": int(block_points.shape[0]),
                }
            )

        for unused_idx in range(args.scale_count, axes_array.size):
            axes_array[unused_idx].set_axis_off()

        fig.suptitle(
            f"Nuage_HP manual block {block_id} ternary PCA across {args.scale_count} scales\n"
            f"block points={block_points.shape[0]}, bbox diagonal={_format_scale(block_sizes[block_id])}",
            fontsize=12,
        )
        output_path = output_dir / f"block_{block_id:02d}_ternary_scales.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

        metadata["plots"][str(block_id)] = {
            "image": str(output_path),
            "scales": block_meta,
        }

    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "scales_m": metadata["scales_m"], "block_count": len(block_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
