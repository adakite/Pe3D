from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.linalg import eig
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError, cKDTree, distance

from .config import Parameters, RunPaths, create_run_paths, load_parameters
from .io import export_colorized_clouds, export_grains, export_granulo, export_smrf_clouds, load_pointcloud
from .math3d import (
    EPS,
    angles_to_reference,
    detrend_quadratic,
    estimate_normals,
    fit_plane,
    normalize,
    orient_vectors,
    rotation_matrix_between,
)
from .plotting import plot_cuboids, plot_elevation, plot_ellipsoid_variants, plot_ellipsoids, plot_granulo, plot_labels


RADFACTOR_CORRECTION = 5.0 / 6.0
MATLAB_DENOISE_NEIGHBOURS = 4
MATLAB_DENOISE_STD_THRESHOLD = 1.0
BLOCK_SCORE_WEIGHTS = np.array(
    [
        -0.023834724887191214,
        0.3749972213604831,
        0.16049088185639013,
        0.1526697997514414,
        0.019609862134321893,
        -0.03296003735371144,
        0.1904630084496496,
    ],
    dtype=np.float64,
)
BLOCK_SCORE_BIAS = -0.09309884486786109
BLOCK_SUPPORT_SCORE_THRESHOLD = 0.48
BLOCK_SEED_SCORE_THRESHOLD = 0.56
BLOCK_GROW_SCORE_THRESHOLD = 0.45
BLOCK_CONNECTION_RADIUS_FACTOR = 2.5
BLOCK_GROW_RADIUS_FACTOR = 1.25
SMRF_GRID_CELL_FACTOR = 3.0
SMRF_MIN_GRID_CELL_SIZE = 0.5
SMRF_MAX_GRID_CELLS = 6_000_000
SMRF_MAX_WINDOW = 12.0
SMRF_HEIGHT_THRESHOLD = 0.35
SMRF_SLOPE_THRESHOLD = 0.75
BLOCK_PCA_MIN_NEIGHBOURS = 6
BLOCK_FEATURE_CHUNK_SIZE = 100_000
BLOCK_SPLIT_COMPONENT_QUANTILE = 0.9997
BLOCK_SPLIT_COMPONENT_MIN_FACTOR = 20.0
BLOCK_SPLIT_MARKER_QUANTILE = 0.92
BLOCK_SPLIT_GRID_CELL_FACTOR = 1.0
BLOCK_SPLIT_SMOOTH_SIGMA = 1.25
BLOCK_SPLIT_PEAK_MERGE_RADIUS_FRACTION = 0.9
BLOCK_SPLIT_MIN_COMPONENT_POINTS = 12
BLOCK_SPLIT_MIN_MARKERS = 2
BLOCK_SPLIT_MAX_MARKERS = 48


@dataclass(slots=True)
class PipelineResult:
    params: Parameters
    run_paths: RunPaths
    csv_path: Path | None
    xlsx_path: Path | None
    colorized_cloud_paths: dict[str, Path]
    labels_count: int
    ellipsoid_count: int
    accepted_ellipsoid_count: int
    sizing_mode: str
    fit_strategy: str
    orientation_mode: str
    preprocess_mode: str
    max_ellipsoid_cloud_fraction: float
    workers: int
    ground_mode: str


def _ellipsoid_is_valid(model: dict[str, object]) -> bool:
    return bool(model.get("fitok")) and bool(model.get("Aqualityok")) and bool(model.get("sizecapok", True))


def _resolve_worker_count(requested_workers: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if requested_workers <= 0:
        return max(1, min(task_count, os.cpu_count() or 1, 8))
    return max(1, min(requested_workers, task_count))


def statistical_denoise(points: np.ndarray, k: int, std_threshold: float) -> np.ndarray:
    if points.shape[0] <= k + 1:
        return np.ones(points.shape[0], dtype=bool)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(k + 1, points.shape[0]), workers=-1)
    neighbour_mean = distances[:, 1:].mean(axis=1)
    threshold = neighbour_mean.mean() + std_threshold * neighbour_mean.std()
    return neighbour_mean <= threshold


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        return points
    voxels = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((counts.size, 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def _multiscale_plan_distance_fast(points: np.ndarray, params: Parameters, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] < 16:
        return np.zeros(points.shape[0], dtype=np.float64)

    tree = cKDTree(points)
    area = max(np.ptp(points[:, 0]) * np.ptp(points[:, 1]), params.minscale * params.minscale)
    max_radius = min(params.maxscale, math.sqrt(area))
    if max_radius <= params.minscale:
        radii = np.array([params.minscale], dtype=np.float64)
    else:
        radii = np.geomspace(params.minscale, max_radius, num=max(params.nscale, 1))

    distances_sum = np.zeros(points.shape[0], dtype=np.float64)
    counts = np.zeros(points.shape[0], dtype=np.float64)
    for radius in radii[::-1]:
        center_count = int(np.clip(area / max(radius * radius, EPS), 32, 512))
        center_indices = rng.integers(0, points.shape[0], size=min(center_count, points.shape[0]))
        neighbourhoods = tree.query_ball_point(points[center_indices], radius, workers=-1)
        for neighbour_idx in neighbourhoods:
            if len(neighbour_idx) < 6:
                continue
            neighbour_idx = np.asarray(neighbour_idx, dtype=np.int64)
            _, _, _, _, distabs = fit_plane(points[neighbour_idx])
            distances_sum[neighbour_idx] += distabs / max(radius, EPS)
            counts[neighbour_idx] += 1.0

    counts[counts == 0.0] = 1.0
    return distances_sum / counts


def _multiscale_plan_distance_matlab(points: np.ndarray, params: Parameters, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] < 3:
        return np.zeros(points.shape[0], dtype=np.float64)

    tree = cKDTree(points)
    area = max(np.ptp(points[:, 0]) * np.ptp(points[:, 1]), params.minscale * params.minscale)
    effective_nscale = max(1, int(round(area)))
    max_radius = min(params.maxscale, math.sqrt(area))
    if max_radius <= params.minscale:
        radii = np.array([params.minscale], dtype=np.float64)
    else:
        radii = np.geomspace(params.minscale, max_radius, num=effective_nscale)
    circle_counts = np.maximum(1, np.rint(effective_nscale / np.square(np.maximum(radii, EPS))).astype(np.int64))

    distances_sum = np.zeros(points.shape[0], dtype=np.float64)
    counts = np.zeros(points.shape[0], dtype=np.float64)
    for radius, circle_count in zip(radii[::-1], circle_counts[::-1], strict=False):
        center_indices = rng.integers(0, points.shape[0], size=int(circle_count))
        unique_centers, repeats = np.unique(center_indices, return_counts=True)
        neighbourhoods = tree.query_ball_point(points[unique_centers], float(radius), workers=-1)
        for repeat_count, neighbour_idx in zip(repeats, neighbourhoods, strict=False):
            if len(neighbour_idx) < 3:
                continue
            neighbour_idx = np.asarray(neighbour_idx, dtype=np.int64)
            _, _, _, signed_distance, _ = fit_plane(points[neighbour_idx])
            weight = float(repeat_count)
            distances_sum[neighbour_idx] += signed_distance * (weight / max(radius, EPS))
            counts[neighbour_idx] += weight

    counts[counts == 0.0] = 1.0
    return distances_sum / counts


def multiscale_plan_distance(
    points: np.ndarray,
    params: Parameters,
    rng: np.random.Generator,
    preprocess_mode: str,
) -> np.ndarray:
    if preprocess_mode == "matlab":
        return _multiscale_plan_distance_matlab(points, params, rng)
    if preprocess_mode == "fast":
        return _multiscale_plan_distance_fast(points, params, rng)
    raise ValueError(f"Unsupported preprocess mode: {preprocess_mode}")


def _build_stacks(receiver: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    count = receiver.size
    donors: list[list[int]] = [[] for _ in range(count)]
    donor_count = np.zeros(count, dtype=np.int64)
    for idx, parent in enumerate(receiver):
        if parent != idx:
            donors[parent].append(idx)
            donor_count[parent] += 1

    sinks = np.flatnonzero(receiver == np.arange(count))
    stacks: list[np.ndarray] = []
    for sink in sinks:
        stack = []
        todo = [int(sink)]
        while todo:
            current = todo.pop()
            stack.append(current)
            todo.extend(donors[current])
        stacks.append(np.asarray(stack, dtype=np.int64))
    return stacks, donor_count, sinks


def _labels_from_stacks(stacks: list[np.ndarray], size: int) -> np.ndarray:
    labels = np.zeros(size, dtype=np.int64)
    for label, stack in enumerate(stacks, start=1):
        labels[stack] = label
    return labels


def segment_labels(points: np.ndarray, neighbour_idx: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    deltas = points[:, None, :] - points[neighbour_idx]
    slope = deltas[:, :, 2] / np.maximum(np.linalg.norm(deltas, axis=2), EPS)
    receiver_choice = np.argmin(slope, axis=1)
    receiver = neighbour_idx[np.arange(points.shape[0]), receiver_choice]
    min_slope = slope[np.arange(points.shape[0]), receiver_choice]
    sinks = np.flatnonzero(min_slope > 0.0)
    receiver[sinks] = sinks
    stacks, donor_count, sinks = _build_stacks(receiver)
    return _labels_from_stacks(stacks, points.shape[0]), stacks, donor_count, sinks


def _border_angle_matrix(
    labels: np.ndarray,
    neighbour_idx: np.ndarray,
    donor_count: np.ndarray,
    normals: np.ndarray,
    nlabels: int,
    nnptcloud: int,
) -> np.ndarray:
    mismatch = nnptcloud - np.sum(labels[neighbour_idx] == labels[:, None], axis=1)
    border_idx = np.flatnonzero((mismatch >= nnptcloud / 4.0) & (donor_count == 0) & (labels > 0))
    angle_sum = np.zeros((nlabels, nlabels), dtype=np.float64)
    angle_count = np.zeros((nlabels, nlabels), dtype=np.float64)
    for idx in border_idx:
        label_i = int(labels[idx])
        neighbours = neighbour_idx[idx]
        neighbour_labels = labels[neighbours]
        valid = neighbour_labels > 0
        if not np.any(valid):
            continue
        angles = angles_to_reference(normals[idx], normals[neighbours[valid]])
        for label_j, angle in zip(neighbour_labels[valid], angles, strict=False):
            angle_sum[label_i - 1, int(label_j) - 1] += float(angle)
            angle_count[label_i - 1, int(label_j) - 1] += 1.0
    return np.divide(angle_sum, angle_count, out=np.zeros_like(angle_sum), where=angle_count > 0)


def _connected_components(adjacency: np.ndarray) -> list[np.ndarray]:
    adjacency = np.asarray(adjacency, dtype=bool)
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, True)
    remaining = np.ones(adjacency.shape[0], dtype=bool)
    components: list[np.ndarray] = []
    while np.any(remaining):
        start = int(np.flatnonzero(remaining)[0])
        todo = [start]
        component = []
        remaining[start] = False
        while todo:
            current = todo.pop()
            component.append(current)
            neighbours = np.flatnonzero(adjacency[current] & remaining)
            remaining[neighbours] = False
            todo.extend(neighbours.tolist())
        components.append(np.asarray(component, dtype=np.int64))
    return components


def _merge_stacks(stacks: list[np.ndarray], components: list[np.ndarray], size: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    merged_stacks: list[np.ndarray] = []
    for component in components:
        merged_stacks.append(np.concatenate([stacks[idx] for idx in component], axis=0))
    labels = _labels_from_stacks(merged_stacks, size)
    sink_idx = np.array([stack[np.argmax(stack)] for stack in merged_stacks], dtype=np.int64)
    return labels, merged_stacks, sink_idx


def cluster_labels(
    points: np.ndarray,
    params: Parameters,
    neighbour_idx: np.ndarray,
    labels: np.ndarray,
    stacks: list[np.ndarray],
    donor_count: np.ndarray,
    sinks: np.ndarray,
    surface: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    nlabels = len(stacks)
    sink_points = points[sinks]
    centre_distance = distance.cdist(sink_points, sink_points)
    area = np.array([surface[stack].sum() for stack in stacks])
    radius = np.sqrt(np.maximum(area, 0.0) / math.pi)
    # Calibrated correction to keep merge behaviour stable across datasets
    # despite slightly different neighbour geometry.
    effective_radfactor = params.radfactor * RADFACTOR_CORRECTION
    merge_distance = effective_radfactor * (radius[:, None] + radius[None, :]) > centre_distance
    neighbour_graph = np.zeros((nlabels, nlabels), dtype=bool)
    for label_idx, stack in enumerate(stacks):
        neighbour_graph[label_idx, np.unique(labels[neighbour_idx[stack]])[np.unique(labels[neighbour_idx[stack]]) > 0] - 1] = True
    border_angles = _border_angle_matrix(labels, neighbour_idx, donor_count, normals, nlabels, params.nnptcloud)
    adjacency = merge_distance & neighbour_graph & (border_angles <= params.maxangle1)
    components = _connected_components(adjacency)
    return _merge_stacks(stacks, components, labels.size)


def clean_labels(
    points: np.ndarray,
    params: Parameters,
    neighbour_idx: np.ndarray,
    labels: np.ndarray,
    stacks: list[np.ndarray],
    donor_count: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    nlabels = len(stacks)
    border_angles = _border_angle_matrix(labels, neighbour_idx, donor_count, normals, nlabels, params.nnptcloud)
    adjacency = (border_angles > 0.0) & (border_angles <= params.maxangle2)
    labels, stacks, _ = _merge_stacks(stacks, _connected_components(adjacency), labels.size)

    filtered_stacks = [stack for stack in stacks if stack.size >= params.minnpoint]
    labels = _labels_from_stacks(filtered_stacks, labels.size)

    final_stacks: list[np.ndarray] = []
    for stack in filtered_stacks:
        centred = points[stack] - points[stack].mean(axis=0)
        _, singular_values, _ = np.linalg.svd(centred, full_matrices=False)
        singular_values = np.pad(singular_values, (0, max(0, 3 - singular_values.size)), constant_values=0.0)
        if singular_values[0] < EPS:
            continue
        keep = (
            singular_values[2] / singular_values[0] > params.minflatness
            or singular_values[1] / singular_values[0] > 2.0 * params.minflatness
        )
        if keep:
            final_stacks.append(stack)

    return _labels_from_stacks(final_stacks, labels.size), final_stacks


def _labels_and_stacks_from_components(components: list[np.ndarray], size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    stacks = [np.asarray(component, dtype=np.int64) for component in components if len(component) > 0]
    return _labels_from_stacks(stacks, size), stacks


def _pca_scale_counts(max_neighbours: int) -> tuple[int, ...]:
    if max_neighbours <= BLOCK_PCA_MIN_NEIGHBOURS:
        return (max_neighbours,)
    candidates = (
        BLOCK_PCA_MIN_NEIGHBOURS,
        max(BLOCK_PCA_MIN_NEIGHBOURS, max_neighbours // 2),
        max_neighbours,
    )
    return tuple(sorted({min(max_neighbours, count) for count in candidates if count >= BLOCK_PCA_MIN_NEIGHBOURS}))


def _normalized_eigenvalues_from_neighbours(neighbour_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centred = neighbour_points - neighbour_points.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / max(neighbour_points.shape[1] - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, EPS, None)
    normalized = eigenvalues / np.maximum(np.sum(eigenvalues, axis=1, keepdims=True), EPS)
    return eigenvalues, normalized


def _local_surface_features(
    points_rot: np.ndarray,
    neighbour_idx: np.ndarray,
    distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if neighbour_idx.size == 0:
        zeros = np.zeros(points_rot.shape[0], dtype=np.float64)
        return zeros, zeros, zeros, zeros, zeros, zeros, zeros, 0.0

    point_count = points_rot.shape[0]
    top_hat = np.zeros(point_count, dtype=np.float64)
    local_relief = np.zeros(point_count, dtype=np.float64)
    local_zstd = np.zeros(point_count, dtype=np.float64)
    scattering = np.zeros(point_count, dtype=np.float64)
    curvature_mean = np.zeros(point_count, dtype=np.float64)
    curvature_max = np.zeros(point_count, dtype=np.float64)
    planarity_mean = np.zeros(point_count, dtype=np.float64)

    scale_counts = _pca_scale_counts(neighbour_idx.shape[1])
    chunk_size = min(BLOCK_FEATURE_CHUNK_SIZE, point_count)
    for start in range(0, point_count, chunk_size):
        stop = min(point_count, start + chunk_size)
        local_neighbour_idx = neighbour_idx[start:stop]
        neighbour_points = points_rot[local_neighbour_idx]
        neighbour_z = neighbour_points[:, :, 2]
        local_low = np.quantile(neighbour_z, 0.15, axis=1)
        local_high = np.quantile(neighbour_z, 0.85, axis=1)
        top_hat[start:stop] = points_rot[start:stop, 2] - local_low
        local_relief[start:stop] = local_high - local_low
        local_zstd[start:stop] = np.std(neighbour_z, axis=1)

        full_eigenvalues, _ = _normalized_eigenvalues_from_neighbours(neighbour_points)
        scattering[start:stop] = full_eigenvalues[:, 0] / full_eigenvalues[:, 2]

        curvature_scales: list[np.ndarray] = []
        planarity_scales: list[np.ndarray] = []
        for count in scale_counts:
            scale_points = neighbour_points[:, :count, :]
            eigenvalues, normalized = _normalized_eigenvalues_from_neighbours(scale_points)
            curvature_scales.append(normalized[:, 0])
            planarity_scales.append((eigenvalues[:, 1] - eigenvalues[:, 0]) / np.maximum(eigenvalues[:, 2], EPS))

        curvature_stack = np.column_stack(curvature_scales)
        planarity_stack = np.column_stack(planarity_scales)
        curvature_mean[start:stop] = np.mean(curvature_stack, axis=1)
        curvature_max[start:stop] = np.max(curvature_stack, axis=1)
        planarity_mean[start:stop] = np.mean(planarity_stack, axis=1)

    if distances.size == 0:
        spacing = 0.0
    else:
        spacing = float(np.median(np.min(distances, axis=1)))
    return top_hat, local_relief, local_zstd, scattering, curvature_mean, curvature_max, planarity_mean, spacing


def _blockiness_score(
    top_hat: np.ndarray,
    local_relief: np.ndarray,
    local_zstd: np.ndarray,
    scattering: np.ndarray,
    curvature_mean: np.ndarray,
    curvature_max: np.ndarray,
    planarity_mean: np.ndarray,
) -> np.ndarray:
    features = np.column_stack(
        [top_hat, local_relief, local_zstd, scattering, curvature_mean, curvature_max, planarity_mean]
    ).astype(np.float64, copy=False)
    centre = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0.0] = 1.0
    standardized = (features - centre) / scale
    logits = standardized @ BLOCK_SCORE_WEIGHTS + BLOCK_SCORE_BIAS
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _robust_unit_interval(values: np.ndarray, *, q_low: float = 0.1, q_high: float = 0.9) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    result = np.zeros_like(values, dtype=np.float64)
    if not np.any(valid):
        return result
    lo = float(np.quantile(values[valid], q_low))
    hi = float(np.quantile(values[valid], q_high))
    if hi <= lo + EPS:
        result[valid] = 1.0
        return result
    result[valid] = np.clip((values[valid] - lo) / (hi - lo), 0.0, 1.0)
    return result


def _bilinear_grid_sample(grid: np.ndarray, min_xy: np.ndarray, cell_size: float, points_xy: np.ndarray) -> np.ndarray:
    if grid.size == 0:
        return np.zeros(points_xy.shape[0], dtype=np.float64)
    ny, nx = grid.shape
    coords = (points_xy - min_xy) / max(cell_size, EPS)
    x = np.clip(coords[:, 0], 0.0, max(nx - 1, 0))
    y = np.clip(coords[:, 1], 0.0, max(ny - 1, 0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    wx = x - x0
    wy = y - y0
    v00 = grid[y0, x0]
    v01 = grid[y1, x0]
    v10 = grid[y0, x1]
    v11 = grid[y1, x1]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + (1.0 - wx) * wy * v01
        + wx * (1.0 - wy) * v10
        + wx * wy * v11
    )


def smrf_ground_model(
    points_rot: np.ndarray,
    distances: np.ndarray,
    *,
    cell_size: float = 0.0,
    max_window: float = SMRF_MAX_WINDOW,
    height_threshold: float = SMRF_HEIGHT_THRESHOLD,
    slope_threshold: float = SMRF_SLOPE_THRESHOLD,
) -> dict[str, np.ndarray | float]:
    if points_rot.size == 0:
        zeros = np.zeros(0, dtype=np.float64)
        return {
            "enabled": 0.0,
            "nonground_mask": np.zeros(0, dtype=bool),
            "height_above_ground": zeros,
            "ground_height": zeros,
            "threshold": zeros,
            "cell_size": 0.0,
            "max_window_cells": 0.0,
            "nonground_fraction": 0.0,
        }

    if distances.size == 0:
        spacing = 0.0
    else:
        spacing = float(np.median(np.min(distances, axis=1)))
    spacing = max(spacing, EPS)

    effective_cell_size = float(cell_size) if cell_size > 0.0 else max(SMRF_GRID_CELL_FACTOR * spacing, SMRF_MIN_GRID_CELL_SIZE)
    points_xy = points_rot[:, :2]
    min_xy = points_xy.min(axis=0)
    extent_xy = np.ptp(points_xy, axis=0)
    estimated_nx = int(np.floor(extent_xy[0] / effective_cell_size)) + 1
    estimated_ny = int(np.floor(extent_xy[1] / effective_cell_size)) + 1
    grid_cells = max(1, estimated_nx * estimated_ny)
    if grid_cells > SMRF_MAX_GRID_CELLS:
        effective_cell_size *= math.sqrt(grid_cells / SMRF_MAX_GRID_CELLS)

    grid_xy = np.floor((points_xy - min_xy) / effective_cell_size).astype(np.int64)
    nx = int(grid_xy[:, 0].max()) + 1
    ny = int(grid_xy[:, 1].max()) + 1
    gx = grid_xy[:, 0]
    gy = grid_xy[:, 1]

    surface = np.full((ny, nx), np.inf, dtype=np.float64)
    np.minimum.at(surface, (gy, gx), points_rot[:, 2])
    occupied = np.isfinite(surface)
    if not np.any(occupied):
        zeros = np.zeros(points_rot.shape[0], dtype=np.float64)
        return {
            "enabled": 0.0,
            "nonground_mask": np.zeros(points_rot.shape[0], dtype=bool),
            "height_above_ground": zeros,
            "ground_height": zeros,
            "threshold": zeros,
            "cell_size": effective_cell_size,
            "max_window_cells": 0.0,
            "nonground_fraction": 0.0,
        }

    nearest_indices = ndimage.distance_transform_edt(~occupied, return_distances=False, return_indices=True)
    surface_filled = surface[tuple(nearest_indices)]

    max_window_cells = max(1, int(round(max_window / max(effective_cell_size, EPS))))
    window_radius = 1
    surface_opened = surface_filled.copy()
    while window_radius <= max_window_cells:
        size = 2 * window_radius + 1
        surface_opened = ndimage.grey_opening(surface_opened, size=(size, size), mode="nearest")
        window_radius *= 2

    grad_y, grad_x = np.gradient(surface_opened, effective_cell_size, effective_cell_size)
    local_slope = np.hypot(grad_x, grad_y)

    ground_height = _bilinear_grid_sample(surface_opened, min_xy, effective_cell_size, points_xy)
    slope_points = _bilinear_grid_sample(local_slope, min_xy, effective_cell_size, points_xy)
    adaptive_threshold = height_threshold + slope_threshold * slope_points * effective_cell_size
    height_above_ground = points_rot[:, 2] - ground_height
    nonground_mask = height_above_ground > adaptive_threshold

    return {
        "enabled": 1.0,
        "nonground_mask": nonground_mask,
        "height_above_ground": height_above_ground,
        "ground_height": ground_height,
        "threshold": adaptive_threshold,
        "cell_size": effective_cell_size,
        "max_window_cells": float(max_window_cells),
        "nonground_fraction": float(np.mean(nonground_mask)),
    }


def _radius_components(points: np.ndarray, radius: float) -> list[np.ndarray]:
    if points.shape[0] == 0:
        return []
    if points.shape[0] == 1:
        return [np.array([0], dtype=np.int64)]

    tree = cKDTree(points)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    parent = np.arange(points.shape[0], dtype=np.int64)
    rank = np.zeros(points.shape[0], dtype=np.int8)

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = int(parent[idx])
        return idx

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for pair in np.asarray(pairs, dtype=np.int64):
        union(int(pair[0]), int(pair[1]))

    roots = np.array([find(idx) for idx in range(points.shape[0])], dtype=np.int64)
    components: list[np.ndarray] = []
    for root in np.unique(roots):
        components.append(np.flatnonzero(roots == root))
    return components


def _split_support_component(
    points_xy: np.ndarray,
    component_indices: np.ndarray,
    seed_mask: np.ndarray,
    score: np.ndarray,
    top_hat: np.ndarray,
    local_relief: np.ndarray,
    spacing: float,
    connection_radius: float,
    min_seed_points: int,
    split_size_threshold: int,
) -> list[np.ndarray]:
    component_indices = np.asarray(component_indices, dtype=np.int64)
    if component_indices.size < max(split_size_threshold, 2 * BLOCK_SPLIT_MIN_COMPONENT_POINTS):
        return [component_indices]

    component_seed_indices = component_indices[seed_mask[component_indices]]
    if component_seed_indices.size < max(2 * min_seed_points, 2 * BLOCK_SPLIT_MIN_COMPONENT_POINTS):
        return [component_indices]

    component_xy = points_xy[component_indices]
    min_xy = component_xy.min(axis=0)
    cell_size = max(BLOCK_SPLIT_GRID_CELL_FACTOR * spacing, EPS)
    grid_xy = np.floor((component_xy - min_xy) / cell_size).astype(np.int64)
    nx = int(grid_xy[:, 0].max()) + 1
    ny = int(grid_xy[:, 1].max()) + 1
    if nx < 2 or ny < 2:
        return [component_indices]

    gx = grid_xy[:, 0]
    gy = grid_xy[:, 1]
    counts = np.zeros((ny, nx), dtype=np.float64)
    score_sum = np.zeros((ny, nx), dtype=np.float64)
    relief_sum = np.zeros((ny, nx), dtype=np.float64)
    top_hat_max = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(counts, (gy, gx), 1.0)
    np.add.at(score_sum, (gy, gx), score[component_indices])
    np.add.at(relief_sum, (gy, gx), local_relief[component_indices])
    np.maximum.at(top_hat_max, (gy, gx), top_hat[component_indices])
    occupied = counts > 0.0
    if np.count_nonzero(occupied) < 4:
        return [component_indices]

    score_mean = np.zeros_like(counts)
    relief_mean = np.zeros_like(counts)
    score_mean[occupied] = score_sum[occupied] / counts[occupied]
    relief_mean[occupied] = relief_sum[occupied] / counts[occupied]

    score_norm = _robust_unit_interval(score_mean[occupied])
    relief_norm = _robust_unit_interval(relief_mean[occupied])
    top_hat_norm = _robust_unit_interval(top_hat_max[occupied])

    block_metric = np.zeros_like(counts)
    block_metric[occupied] = 0.60 * score_norm + 0.25 * top_hat_norm + 0.15 * relief_norm

    smooth_num = ndimage.gaussian_filter(block_metric * counts, sigma=BLOCK_SPLIT_SMOOTH_SIGMA, mode="nearest")
    smooth_den = ndimage.gaussian_filter(counts, sigma=BLOCK_SPLIT_SMOOTH_SIGMA, mode="nearest")
    smooth_metric = np.zeros_like(counts)
    smooth_metric[occupied] = smooth_num[occupied] / np.maximum(smooth_den[occupied], EPS)
    smooth_metric[~occupied] = 0.0

    marker_threshold = float(np.quantile(smooth_metric[occupied], BLOCK_SPLIT_MARKER_QUANTILE))
    local_max = ndimage.maximum_filter(smooth_metric, size=3, mode="nearest")
    marker_mask = occupied & (smooth_metric >= marker_threshold) & (smooth_metric >= local_max - 1e-9)
    if np.count_nonzero(marker_mask) < BLOCK_SPLIT_MIN_MARKERS:
        return [component_indices]

    peak_coords = np.column_stack(np.nonzero(marker_mask)).astype(np.float64)
    peak_merge_radius = max(1.0, BLOCK_SPLIT_PEAK_MERGE_RADIUS_FRACTION * connection_radius / cell_size)
    peak_components_local = _radius_components(peak_coords, peak_merge_radius)

    marker_strengths: list[tuple[float, int]] = []
    for component_id, component in enumerate(peak_components_local):
        coords = peak_coords[np.asarray(component, dtype=np.int64)].astype(np.int64)
        marker_strengths.append((float(np.max(smooth_metric[coords[:, 0], coords[:, 1]])), component_id))
    if len(marker_strengths) < BLOCK_SPLIT_MIN_MARKERS:
        return [component_indices]
    marker_strengths.sort(reverse=True)
    selected_markers = [marker_id for _, marker_id in marker_strengths[:BLOCK_SPLIT_MAX_MARKERS]]

    markers_filtered = np.zeros_like(counts, dtype=np.int32)
    for new_id, marker_id in enumerate(selected_markers, start=1):
        coords = peak_coords[np.asarray(peak_components_local[marker_id], dtype=np.int64)].astype(np.int64)
        markers_filtered[coords[:, 0], coords[:, 1]] = new_id
    if int(markers_filtered.max()) < BLOCK_SPLIT_MIN_MARKERS:
        return [component_indices]

    cost = np.full_like(markers_filtered, 255, dtype=np.uint8)
    cost[occupied] = np.clip(np.rint(255.0 * (1.0 - smooth_metric[occupied])), 0.0, 255.0).astype(np.uint8)
    flooded = ndimage.watershed_ift(cost, markers_filtered.astype(np.int32))
    flooded[~occupied] = 0

    assigned_labels = flooded[gy, gx]
    if np.count_nonzero(assigned_labels > 0) < max(2 * min_seed_points, BLOCK_SPLIT_MIN_COMPONENT_POINTS):
        return [component_indices]

    split_components: list[np.ndarray] = []
    min_component_size = max(BLOCK_SPLIT_MIN_COMPONENT_POINTS, min_seed_points * 2)
    for label in np.unique(assigned_labels[assigned_labels > 0]):
        assigned = component_indices[assigned_labels == label]
        if assigned.size >= min_component_size and np.count_nonzero(seed_mask[assigned]) >= min_seed_points:
            split_components.append(assigned)
    if len(split_components) < 2:
        return [component_indices]
    return split_components


def segment_blocks(
    points: np.ndarray,
    points_rot: np.ndarray,
    params: Parameters,
    neighbour_idx: np.ndarray,
    distances: np.ndarray,
    *,
    enable_split: bool = False,
    prefilter_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, float]]:
    (
        top_hat,
        local_relief,
        local_zstd,
        scattering,
        curvature_mean,
        curvature_max,
        planarity_mean,
        spacing,
    ) = _local_surface_features(points_rot, neighbour_idx, distances)
    score = _blockiness_score(
        top_hat,
        local_relief,
        local_zstd,
        scattering,
        curvature_mean,
        curvature_max,
        planarity_mean,
    )
    support_mask = score >= BLOCK_SUPPORT_SCORE_THRESHOLD
    if prefilter_mask is not None:
        support_mask &= np.asarray(prefilter_mask, dtype=bool)
    if not np.any(support_mask):
        return np.zeros(points.shape[0], dtype=np.int64), [], {"spacing": spacing}

    spacing = max(spacing, 1e-6)
    seed_mask = score >= BLOCK_SEED_SCORE_THRESHOLD
    if prefilter_mask is not None:
        seed_mask &= np.asarray(prefilter_mask, dtype=bool)
    support_indices = np.flatnonzero(support_mask)
    if support_indices.size == 0:
        return np.zeros(points.shape[0], dtype=np.int64), [], {"spacing": spacing}

    connection_radius = max(BLOCK_CONNECTION_RADIUS_FACTOR * spacing, 1.5 * spacing)
    horizontal_points = points_rot[:, :2]
    split_grid_cell_size = max(BLOCK_SPLIT_GRID_CELL_FACTOR * spacing, EPS)
    support_components_local = _radius_components(points_rot[support_indices], connection_radius)
    min_seed_points = max(3, params.nnptcloud // 4)
    eligible_support_components: list[np.ndarray] = []
    for component in support_components_local:
        component_indices = support_indices[component]
        if component_indices.size < params.minnpoint:
            continue
        if np.count_nonzero(seed_mask[component_indices]) < min_seed_points:
            continue
        eligible_support_components.append(component_indices)

    eligible_sizes = np.array([component.size for component in eligible_support_components], dtype=np.int64)
    split_size_threshold = int(
        max(
            BLOCK_SPLIT_COMPONENT_MIN_FACTOR * params.minnpoint,
            np.quantile(eligible_sizes, BLOCK_SPLIT_COMPONENT_QUANTILE) if eligible_sizes.size else 0.0,
        )
    )

    kept_support_components: list[np.ndarray] = []
    split_attempt_count = 0
    for component_indices in eligible_support_components:
        if enable_split and component_indices.size >= split_size_threshold:
            split_attempt_count += 1
            kept_support_components.extend(
                _split_support_component(
                    horizontal_points,
                    component_indices,
                    seed_mask,
                    score,
                    top_hat,
                    local_relief,
                    spacing,
                    connection_radius,
                    min_seed_points,
                    split_size_threshold,
                )
            )
        else:
            kept_support_components.append(component_indices)

    if not kept_support_components:
        return np.zeros(points.shape[0], dtype=np.int64), [], {"spacing": spacing}

    candidate_indices = np.concatenate(kept_support_components)
    candidate_tree = cKDTree(points_rot[candidate_indices])
    candidate_labels, _ = _labels_and_stacks_from_components(kept_support_components, points.shape[0])
    candidate_component_labels = candidate_labels[candidate_indices]

    grow_radius = BLOCK_GROW_RADIUS_FACTOR * connection_radius
    nearest_distance, nearest_idx = candidate_tree.query(points_rot, k=1, workers=-1)
    grow_mask = (score >= BLOCK_GROW_SCORE_THRESHOLD) & (nearest_distance <= grow_radius)
    if prefilter_mask is not None:
        grow_mask &= np.asarray(prefilter_mask, dtype=bool)

    final_labels = np.zeros(points.shape[0], dtype=np.int64)
    final_labels[grow_mask] = candidate_component_labels[np.asarray(nearest_idx[grow_mask], dtype=np.int64)]
    final_labels[candidate_indices] = candidate_component_labels

    stacks = [np.flatnonzero(final_labels == label) for label in np.unique(final_labels[final_labels > 0])]
    filtered_stacks: list[np.ndarray] = []
    for stack in stacks:
        if stack.size < params.minnpoint:
            continue
        centred = points[stack] - points[stack].mean(axis=0)
        _, singular_values, _ = np.linalg.svd(centred, full_matrices=False)
        singular_values = np.pad(singular_values, (0, max(0, 3 - singular_values.size)), constant_values=0.0)
        if singular_values[0] < EPS:
            continue
        if singular_values[2] / singular_values[0] <= 0.02 and singular_values[1] / singular_values[0] <= 0.08:
            continue
        filtered_stacks.append(stack)

    final_labels = _labels_from_stacks(filtered_stacks, points.shape[0])
    diagnostics = {
        "spacing": spacing,
        "block_split_enabled": float(bool(enable_split)),
        "support_score_threshold": BLOCK_SUPPORT_SCORE_THRESHOLD,
        "seed_score_threshold": BLOCK_SEED_SCORE_THRESHOLD,
        "grow_score_threshold": BLOCK_GROW_SCORE_THRESHOLD,
        "prefilter_points": float(np.count_nonzero(prefilter_mask)) if prefilter_mask is not None else float(points.shape[0]),
        "connection_radius": connection_radius,
        "grow_radius": grow_radius,
        "split_seed_radius": split_grid_cell_size if enable_split else 0.0,
        "split_size_threshold": float(split_size_threshold),
        "split_attempt_count": float(split_attempt_count),
        "candidate_points": float(candidate_indices.size),
        "split_component_count": float(len(kept_support_components)),
    }
    return final_labels, filtered_stacks, diagnostics


def fit_pca_cuboid(points: np.ndarray) -> dict[str, np.ndarray]:
    if points.shape[0] == 0:
        zeros = np.zeros((8, 3), dtype=np.float64)
        return {"center": np.zeros(3), "extents": np.zeros(3), "rotation": np.eye(3), "corners": zeros}
    if points.shape[0] == 1:
        corners = np.repeat(points, 8, axis=0)
        return {"center": points[0], "extents": np.zeros(3), "rotation": np.eye(3), "corners": corners}
    if points.shape[0] == 2:
        centre = points.mean(axis=0)
        extent = np.abs(points[1] - points[0])
        corners = np.repeat(points[[0]], 8, axis=0)
        corners[4:] = points[1]
        return {"center": centre, "extents": extent, "rotation": np.eye(3), "corners": corners}

    centre = points.mean(axis=0)
    cov = np.cov((points - centre).T)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    local = (points - centre) @ rotation
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    local_centre = (mins + maxs) / 2.0
    extents = maxs - mins
    offsets = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ]
    )
    return {
        "center": centre + local_centre @ rotation.T,
        "extents": extents,
        "rotation": rotation,
        "corners": offsets @ rotation.T + centre,
    }


def _sorted_cuboid_axes(cuboid: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    extents = np.asarray(cuboid["extents"], dtype=np.float64)
    rotation = np.asarray(cuboid["rotation"], dtype=np.float64)
    order = np.argsort(extents)[::-1]
    extents = extents[order]
    rotation = rotation[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    return extents, rotation


def _fit_block_model(
    points: np.ndarray,
    cuboid: dict[str, np.ndarray],
    max_major_diameter: float | None,
) -> dict[str, object]:
    extents, rotation = _sorted_cuboid_axes(cuboid)
    radii = np.maximum(0.5 * extents, 1e-4)
    centre = np.asarray(cuboid["center"], dtype=np.float64)
    obb_volume = float(np.prod(extents))
    obb_area = float(2.0 * (extents[0] * extents[1] + extents[0] * extents[2] + extents[1] * extents[2]))

    volume = obb_volume
    area = obb_area
    if points.shape[0] >= 4:
        try:
            hull = ConvexHull(points, qhull_options="QJ")
            volume = float(hull.volume)
            area = float(hull.area)
        except QhullError:
            pass
        except Exception:
            pass

    major_diameter = float(extents[0])
    occupancy = 100.0 * volume / max(obb_volume, EPS)
    sizecapok = max_major_diameter is None or major_diameter <= max_major_diameter + 1e-9
    model: dict[str, object] = {
        "fitok": True,
        "Aqualityok": True,
        "sizecapok": sizecapok,
        "c": centre,
        "r": radii,
        "R": rotation,
        "axis1": rotation[:, 0],
        "axis2": rotation[:, 1],
        "axis3": rotation[:, 2],
        "axis_map": rotation[:, 0],
        "axis_x": rotation[:, 0],
        "d": np.zeros(points.shape[0], dtype=np.float64),
        "r2": 1.0,
        "V": volume,
        "A": area,
        "major_diameter": major_diameter,
        "max_major_diameter": max_major_diameter,
        "Aratio": volume / max(obb_volume, EPS),
        "Acover": min(max(occupancy, 0.0), 100.0),
        "quality_label": "obb_occupancy_percent",
    }
    return model


def _surface_area_ellipsoid(radii: np.ndarray) -> float:
    power = 1.6075
    a, b, c = radii
    return float(4.0 * math.pi * (((a * b) ** power + (a * c) ** power + (b * c) ** power) / 3.0) ** (1.0 / power))


def _random_ellipsoid_surface_samples(
    radii: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * math.pi, size=samples)
    z = rng.uniform(-1.0, 1.0, size=samples)
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    sphere = np.column_stack([radial * np.cos(theta), radial * np.sin(theta), z])
    return (sphere * radii) @ rotation.T + center


def _finalize_ellipsoid_model(
    points: np.ndarray,
    surface_sum: float,
    params: Parameters,
    center: np.ndarray,
    radii: np.ndarray,
    rotation: np.ndarray,
    rng: np.random.Generator,
    max_major_diameter: float | None,
    *,
    axis_map: np.ndarray | None = None,
    axis_x: np.ndarray | None = None,
) -> dict[str, object]:
    radii = np.asarray(radii, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    rotation = rotation[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0

    local = (points - center) @ rotation
    scale = np.sqrt(np.sum((local / radii) ** 2, axis=1))
    scale = np.maximum(scale, EPS)
    nearest_local = local / scale[:, None]
    distances = np.linalg.norm(local - nearest_local, axis=1)
    radial_residual = scale - 1.0
    ss_res = float(np.sum(radial_residual * radial_residual))
    ss_tot = float(np.sum((scale - scale.mean()) ** 2)) + EPS

    volume = float((4.0 / 3.0) * math.pi * np.prod(radii))
    area = _surface_area_ellipsoid(radii)
    major_diameter = float(2.0 * radii[0])
    sample_points = _random_ellipsoid_surface_samples(radii, center, rotation, 200, rng)
    sample_tree = cKDTree(sample_points)
    _, matched = sample_tree.query(points, k=1)
    acover = 100.0 * np.unique(matched).size / max(sample_points.shape[0], 1)
    sizecapok = max_major_diameter is None or major_diameter <= max_major_diameter + 1e-9

    model: dict[str, object] = {
        "fitok": True,
        "Aqualityok": acover > params.aquality_thresh,
        "sizecapok": sizecapok,
        "c": center,
        "r": radii,
        "R": rotation,
        "axis1": rotation[:, 0],
        "axis2": rotation[:, 1],
        "axis3": rotation[:, 2],
        "axis_map": axis_map if axis_map is not None else rotation[:, 0],
        "axis_x": axis_x if axis_x is not None else rotation[:, 0],
        "d": distances,
        "r2": 1.0 - ss_res / ss_tot,
        "V": volume,
        "A": area,
        "major_diameter": major_diameter,
        "max_major_diameter": max_major_diameter,
        "Aratio": surface_sum / max(area, EPS),
        "Acover": acover,
    }
    return model


def _fit_direct_algebraic(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if points.shape[0] < 9:
        return None
    mean = points.mean(axis=0)
    scale = 1.0 / max(float(np.max(points.max(axis=0) - points.min(axis=0))), 1e-8)
    x = scale * (points[:, 0] - mean[0])
    y = scale * (points[:, 1] - mean[1])
    z = scale * (points[:, 2] - mean[2])

    design = np.column_stack([x * x, y * y, z * z, 2 * y * z, 2 * x * z, 2 * x * y, 2 * x, 2 * y, 2 * z, np.ones_like(x)])
    scatter = design.T @ design
    k = 4.0
    c1 = np.array([[0.0, k, k], [k, 0.0, k], [k, k, 0.0]], dtype=np.float64) / 2.0 - 1.0
    c2 = -k * np.eye(3, dtype=np.float64)
    constraint = np.block([[c1, np.zeros((3, 3))], [np.zeros((3, 3)), c2]])
    s11 = scatter[:6, :6]
    s12 = scatter[:6, 6:]
    s22 = scatter[6:, 6:]

    try:
        reduced = s11 - s12 @ np.linalg.solve(s22, s12.T)
        eigenvalues, eigenvectors = eig(reduced, constraint)
        eigenvalues = np.real_if_close(eigenvalues, tol=1000)
        eigenvectors = np.real_if_close(eigenvectors, tol=1000)
        valid = np.isfinite(eigenvalues) & (np.abs(np.imag(eigenvalues)) < 1e-8)
        eigenvalues = np.real(eigenvalues[valid])
        eigenvectors = np.real(eigenvectors[:, valid])
        positive = eigenvalues > 0
        if np.any(positive):
            v1 = eigenvectors[:, np.flatnonzero(positive)[0]]
        else:
            v1 = eigenvectors[:, np.argmin(np.abs(eigenvalues))]
        v2 = -np.linalg.solve(s22, s12.T @ v1)
        vector = np.concatenate([v1, v2])
    except Exception:
        return None

    params = np.zeros(10, dtype=np.float64)
    params[:3] = vector[:3]
    params[3:6] = 2.0 * vector[5:2:-1]
    params[6:9] = 2.0 * vector[6:9]
    params[9] = vector[9]

    params_half = params.copy()
    params_half[3:9] *= 0.5
    matrix = np.array(
        [
            [params_half[0], params_half[3], params_half[4], params_half[6]],
            [params_half[3], params_half[1], params_half[5], params_half[7]],
            [params_half[4], params_half[5], params_half[2], params_half[8]],
            [params_half[6], params_half[7], params_half[8], params_half[9]],
        ],
        dtype=np.float64,
    )
    try:
        center = np.linalg.solve(matrix[:3, :3], -params_half[6:9])
    except np.linalg.LinAlgError:
        return None
    transform = np.eye(4, dtype=np.float64)
    transform[3, :3] = center
    translated = transform @ matrix @ transform.T
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(translated[:3, :3])
    except np.linalg.LinAlgError:
        return None
    if np.any(eigenvalues <= 0) or -translated[3, 3] <= 0:
        return None
    radii = np.sqrt((-translated[3, 3]) / eigenvalues)
    order = np.argsort(radii)[::-1]
    radii = radii[order] / scale
    rotation = eigenvectors[:, order]
    center = center / scale + mean
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    return center, radii, rotation


def _fit_ellipsoid(
    points: np.ndarray,
    surface_sum: float,
    params: Parameters,
    rng: np.random.Generator,
    strategy: str,
    max_major_diameter: float | None,
) -> dict[str, object]:
    model: dict[str, object] = {"fitok": False, "Aqualityok": False, "sizecapok": True}
    if points.shape[0] < 3:
        return model

    centre = points.mean(axis=0)
    cov = np.cov((points - centre).T)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0
    local = (points - centre) @ rotation
    radii = np.maximum(np.max(np.abs(local), axis=0), 1e-4)

    if strategy in {"bounded_pca", "hybrid_direct"} and points.shape[0] >= 9:
        sample = points if points.shape[0] <= 300 else points[rng.choice(points.shape[0], size=300, replace=False)]

        def residuals(theta: np.ndarray) -> np.ndarray:
            delta_centre = theta[:3]
            log_scale = theta[3:]
            trial_centre = centre + delta_centre @ rotation.T
            trial_radii = radii * np.exp(log_scale)
            local_sample = (sample - trial_centre) @ rotation
            return np.sum((local_sample / trial_radii) ** 2, axis=1) - 1.0

        lower = np.array(
            [
                -0.25 * radii[0],
                -0.25 * radii[1],
                -0.25 * radii[2],
                math.log(0.5),
                math.log(0.5),
                math.log(0.5),
            ]
        )
        upper = np.array(
            [
                0.25 * radii[0],
                0.25 * radii[1],
                0.25 * radii[2],
                math.log(1.5),
                math.log(1.5),
                math.log(1.5),
            ]
        )
        try:
            refined = least_squares(
                residuals,
                np.zeros(6, dtype=np.float64),
                bounds=(lower, upper),
                loss="soft_l1",
                max_nfev=120,
            )
            centre = centre + refined.x[:3] @ rotation.T
            radii = radii * np.exp(refined.x[3:])
        except Exception:
            pass

    pca_model = _finalize_ellipsoid_model(points, surface_sum, params, centre, radii, rotation, rng, max_major_diameter)
    if strategy != "hybrid_direct":
        return pca_model

    direct_fit = _fit_direct_algebraic(points)
    if direct_fit is None:
        return pca_model

    direct_center, direct_radii, direct_rotation = direct_fit
    # The original MATLAB direct fit applies the 4J - I^2 > 0 constraint,
    # which keeps the smallest radius from collapsing too far below the
    # largest one. Mirror that minimum-axis behaviour here before the model
    # is finalized.
    direct_radii = np.asarray(direct_radii, dtype=np.float64).copy()
    direct_radii[2] = max(direct_radii[2], 0.5 * direct_radii[0])
    direct_radii[1] = max(direct_radii[1], direct_radii[2])
    # The map-view angle is more stable with a row-based axis convention for
    # the direct algebraic basis than with the leading column alone.
    direct_axis_map = direct_rotation[0, :].copy()
    hybrid_model = _finalize_ellipsoid_model(
        points,
        surface_sum,
        params,
        direct_center,
        direct_radii,
        direct_rotation,
        rng,
        max_major_diameter,
        axis_map=direct_axis_map,
        axis_x=pca_model["axis_x"],
    )
    if _ellipsoid_is_valid(hybrid_model):
        return hybrid_model
    return pca_model


def _fit_grain_models(task: tuple[np.ndarray, float, Parameters, str, str, float | None, int]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    grain_points, surface_sum, params, fit_strategy, sizing_mode, max_major_diameter, rng_seed = task
    rng = np.random.default_rng(rng_seed)
    cuboid = fit_pca_cuboid(grain_points)
    if sizing_mode == "block":
        model = _fit_block_model(grain_points, cuboid, max_major_diameter)
    else:
        model = _fit_ellipsoid(grain_points, surface_sum, params, rng, fit_strategy, max_major_diameter)
    return cuboid, model


def _iter_fit_tasks(
    points: np.ndarray,
    surface: np.ndarray,
    stacks: list[np.ndarray],
    params: Parameters,
    fit_strategy: str,
    sizing_mode: str,
    max_major_diameter: float | None,
    seed: int,
):
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(len(stacks))
    for stack, child_sequence in zip(stacks, child_sequences, strict=False):
        grain_points = points[stack]
        surface_sum = float(surface[stack].sum())
        rng_seed = int(child_sequence.generate_state(1, dtype=np.uint64)[0])
        yield (grain_points, surface_sum, params, fit_strategy, sizing_mode, max_major_diameter, rng_seed)


def _empty_granulo() -> dict[str, np.ndarray | int]:
    return {
        "diameter": np.empty((3, 0)),
        "diameter_edges_lin": np.array([0.0, 1.0]),
        "diameter_edges_log": np.array([1e-6, 1e-5]),
        "vol": np.empty(0),
        "vol_edges_lin": np.array([0.0, 1.0]),
        "vol_edges_log": np.array([1e-6, 1e-5]),
        "area": np.empty(0),
        "area_edges_lin": np.array([0.0, 1.0]),
        "area_edges_log": np.array([1e-6, 1e-5]),
        "Acover": np.empty(0),
        "r2": np.empty(0),
        "d": np.empty(0),
        "nbin": 1,
        "valid_indices": np.empty(0, dtype=np.int64),
        "Acover_label": "surface cover (%)",
    }


def grain_size_distribution(ellipsoids: list[dict[str, object]]) -> dict[str, np.ndarray | int]:
    valid_indices = np.array([idx for idx, item in enumerate(ellipsoids) if _ellipsoid_is_valid(item)], dtype=np.int64)
    if valid_indices.size == 0:
        return _empty_granulo()

    radii = np.column_stack([np.asarray(ellipsoids[idx]["r"], dtype=np.float64) for idx in valid_indices])
    diameter = 2.0 * radii
    nbin = int(math.ceil(math.sqrt(diameter.shape[1])))
    nbin = max(nbin, 2)

    diameter_min = max(float(np.min(diameter[2])), 1e-6)
    diameter_max = max(float(np.max(diameter[0])), diameter_min * 1.01)
    diameter_edges_lin = np.linspace(diameter_min, diameter_max, nbin)
    diameter_edges_log = np.geomspace(diameter_min, diameter_max, nbin)

    volume = np.array([float(ellipsoids[idx]["V"]) for idx in valid_indices], dtype=np.float64)
    area = np.array([float(ellipsoids[idx]["A"]) for idx in valid_indices], dtype=np.float64)

    volume_min = max(float(np.min(volume)), 1e-9)
    volume_max = max(float(np.max(volume)), volume_min * 1.01)
    area_min = max(float(np.min(area)), 1e-9)
    area_max = max(float(np.max(area)), area_min * 1.01)

    granulo = {
        "diameter": diameter,
        "diameter_edges_lin": diameter_edges_lin,
        "diameter_edges_log": diameter_edges_log,
        "vol": volume,
        "vol_edges_lin": np.linspace(volume_min, volume_max, nbin),
        "vol_edges_log": np.geomspace(volume_min, volume_max, nbin),
        "area": area,
        "area_edges_lin": np.linspace(area_min, area_max, nbin),
        "area_edges_log": np.geomspace(area_min, area_max, nbin),
        "Acover": np.array([float(ellipsoids[idx]["Acover"]) for idx in valid_indices], dtype=np.float64),
        "r2": np.array([float(ellipsoids[idx]["r2"]) for idx in valid_indices], dtype=np.float64),
        "d": np.array([float(np.mean(np.asarray(ellipsoids[idx]["d"], dtype=np.float64))) for idx in valid_indices], dtype=np.float64),
        "nbin": nbin,
        "valid_indices": valid_indices,
        "Acover_label": str(ellipsoids[int(valid_indices[0])].get("quality_label", "surface cover (%)")),
    }
    return granulo


def _orientation_angle(axis: np.ndarray, mode: str, x_view: bool = False) -> float:
    if x_view:
        numerator = axis[1]
        denominator = axis[2]
    else:
        numerator = axis[1]
        denominator = axis[0]

    if mode == "matlab":
        if abs(denominator) < 1e-12:
            angle = math.pi / 2.0
        else:
            angle = math.atan(numerator / denominator) + math.pi / 2.0
    else:
        angle = math.atan2(numerator, denominator) + math.pi / 2.0
    return angle % math.pi


def ellipsoid_orientation(
    points_rot: np.ndarray,
    ellipsoids: list[dict[str, object]],
    granulo: dict[str, np.ndarray | int],
    orientation_mode: str,
) -> dict[str, np.ndarray | int]:
    valid_indices = np.asarray(granulo["valid_indices"], dtype=np.int64)
    if valid_indices.size == 0:
        granulo.update(
            {
                "angle_Mview": np.empty(0),
                "angle_Xview": np.empty(0),
                "u_Mview": np.empty(0),
                "v_Mview": np.empty(0),
                "w_Mview": np.empty(0),
                "u_Xview": np.empty(0),
                "v_Xview": np.empty(0),
                "w_Xview": np.empty(0),
                "radius": np.empty(0),
                "Location": np.empty((3, 0)),
            }
        )
        return granulo

    delta = 1e32
    sensor_map = np.array([points_rot[:, 0].mean(), points_rot[:, 1].mean() + delta, points_rot[:, 2].mean()])
    sensor_x = np.array([points_rot[:, 0].mean(), points_rot[:, 1].mean(), points_rot[:, 2].mean() + delta])

    u_map = []
    v_map = []
    w_map = []
    angle_map = []
    u_x = []
    v_x = []
    w_x = []
    angle_x = []
    radius = []
    locations = []
    for idx in valid_indices:
        ellipsoid = ellipsoids[int(idx)]
        axis = np.asarray(ellipsoid.get("axis_map", ellipsoid["axis1"]), dtype=np.float64).copy()
        if sensor_map @ axis < 0.0:
            axis *= -1.0
        u_map.append(axis[0])
        v_map.append(axis[1])
        w_map.append(axis[2])
        angle_map.append(_orientation_angle(axis, orientation_mode, x_view=False))

        axis_x = np.asarray(ellipsoid.get("axis_x", ellipsoid["axis1"]), dtype=np.float64).copy()
        if sensor_x @ axis_x < 0.0:
            axis_x *= -1.0
        u_x.append(axis_x[0])
        v_x.append(axis_x[1])
        w_x.append(axis_x[2])
        angle_x.append(_orientation_angle(axis_x, orientation_mode, x_view=True))

        radius.append(float(ellipsoid["r"][0]))
        locations.append(np.asarray(ellipsoid["c"], dtype=np.float64))

    granulo.update(
        {
            "angle_Mview": np.asarray(angle_map, dtype=np.float64),
            "angle_Xview": np.asarray(angle_x, dtype=np.float64),
            "u_Mview": np.asarray(u_map, dtype=np.float64),
            "v_Mview": np.asarray(v_map, dtype=np.float64),
            "w_Mview": np.asarray(w_map, dtype=np.float64),
            "u_Xview": np.asarray(u_x, dtype=np.float64),
            "v_Xview": np.asarray(v_x, dtype=np.float64),
            "w_Xview": np.asarray(w_x, dtype=np.float64),
            "radius": np.asarray(radius, dtype=np.float64),
            "Location": np.column_stack(locations),
        }
    )
    return granulo


def _plot_outputs(
    points: np.ndarray,
    points_rot: np.ndarray,
    initial_labels: np.ndarray,
    clustered_labels: np.ndarray,
    cleaned_labels: np.ndarray | None,
    cuboids: list[dict[str, np.ndarray]],
    models: list[dict[str, object]],
    granulo: dict[str, np.ndarray | int],
    run_paths: RunPaths,
    sizing_mode: str,
) -> None:
    plot_elevation(points_rot, run_paths.figure_dir / "elevation")
    plot_labels(points, initial_labels, run_paths.figure_dir / "labels_ini")
    plot_labels(points, clustered_labels, run_paths.figure_dir / "labels_cluster")
    if cleaned_labels is not None:
        plot_labels(points, cleaned_labels, run_paths.figure_dir / "labels_clean")
    plot_cuboids(points, cuboids, run_paths.figure_dir / "fitted_cuboids")
    if sizing_mode == "ellipsoid":
        plot_ellipsoids(points, models, run_paths.figure_dir / "fitted_ellipsoids")
        plot_ellipsoid_variants(points, models, granulo, run_paths.figure_dir)
    plot_granulo(granulo, run_paths.figure_dir)


def run_pipeline(
    pointcloud_path: str | Path,
    *,
    param_csv: str | Path | None = None,
    output_root: str | Path = "pe3d_outputs",
    save_plots: bool | None = None,
    save_grains: bool | None = None,
    save_colored_clouds: bool = False,
    seed: int = 0,
    sizing_mode: str = "ellipsoid",
    fit_strategy: str = "bounded_pca",
    orientation_mode: str = "atan2",
    preprocess_mode: str = "matlab",
    max_ellipsoid_cloud_fraction: float = 0.25,
    workers: int = 1,
    block_split: bool = False,
    ground_mode: str = "none",
    smrf_cell_size: float = 0.0,
    smrf_max_window: float = SMRF_MAX_WINDOW,
    smrf_height_threshold: float = SMRF_HEIGHT_THRESHOLD,
    smrf_slope_threshold: float = SMRF_SLOPE_THRESHOLD,
) -> PipelineResult:
    rng = np.random.default_rng(seed)
    params = load_parameters(pointcloud_path, param_csv=param_csv)
    run_paths = create_run_paths(output_root, params.pointcloud_name)
    cloud = load_pointcloud(params.pointcloud_path)
    params.iscolor = cloud.iscolor

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
    cloud_major_span = float(np.max(np.ptp(points, axis=0))) if points.shape[0] else 0.0
    max_major_diameter = None
    if max_ellipsoid_cloud_fraction > 0.0 and cloud_major_span > 0.0:
        max_major_diameter = float(max_ellipsoid_cloud_fraction * cloud_major_span)

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
    surface = math.pi * np.square(np.maximum(np.min(distances, axis=1), 0.0))
    smrf_result: dict[str, np.ndarray | float] | None = None
    if sizing_mode == "block":
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

    cleaned_labels = None
    if sizing_mode == "block":
        initial_labels, stacks, _ = segment_blocks(
            points,
            points_rot,
            params,
            neighbour_idx,
            distances,
            enable_split=block_split,
            prefilter_mask=None if smrf_result is None else np.asarray(smrf_result["nonground_mask"], dtype=bool),
        )
        clustered_labels = initial_labels
        final_labels = initial_labels
        final_stacks = stacks
    else:
        sensor_center = np.array([points[:, 0].mean(), points[:, 1].mean(), 1e32])
        normals = estimate_normals(points, params.nnptcloud, sensor_center=sensor_center)

        initial_labels, stacks, donor_count, sinks = segment_labels(points_rot, neighbour_idx)
        clustered_labels, stacks, _ = cluster_labels(
            points,
            params,
            neighbour_idx,
            initial_labels,
            stacks,
            donor_count,
            sinks,
            surface,
            normals,
        )

        final_labels = clustered_labels
        final_stacks = stacks
        if params.clean:
            cleaned_labels, final_stacks = clean_labels(
                points,
                params,
                neighbour_idx,
                clustered_labels,
                stacks,
                donor_count,
                normals,
            )
            final_labels = cleaned_labels

    fit_tasks = _iter_fit_tasks(points, surface, final_stacks, params, fit_strategy, sizing_mode, max_major_diameter, seed)
    effective_workers = _resolve_worker_count(workers, len(final_stacks))
    if effective_workers == 1:
        fitted_models = [_fit_grain_models(task) for task in fit_tasks]
    else:
        with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="pe3d-fit") as executor:
            fitted_models = list(executor.map(_fit_grain_models, fit_tasks))
    cuboids = [item[0] for item in fitted_models]
    models = [item[1] for item in fitted_models]
    granulo = ellipsoid_orientation(points_rot, models, grain_size_distribution(models), orientation_mode)

    csv_path = None
    xlsx_path = None
    colorized_cloud_paths: dict[str, Path] = {}
    if save_grains if save_grains is not None else params.savegrain:
        export_grains(points, final_labels, run_paths, params.pointcloud_name)
    if save_colored_clouds:
        colorized_cloud_paths = export_colorized_clouds(
            raw_points,
            points,
            final_labels,
            models,
            granulo,
            run_paths,
            params.pointcloud_name,
        )
        if smrf_result is not None:
            colorized_cloud_paths.update(
                export_smrf_clouds(
                    raw_points,
                    points,
                    np.asarray(smrf_result["height_above_ground"], dtype=np.float64),
                    np.asarray(smrf_result["nonground_mask"], dtype=bool),
                    run_paths,
                    params.pointcloud_name,
                )
            )
    if params.savegranulo:
        csv_path, xlsx_path = export_granulo(run_paths, params.pointcloud_name, granulo)
    if save_plots if save_plots is not None else params.saveplot:
        _plot_outputs(points, points_rot, initial_labels, clustered_labels, cleaned_labels, cuboids, models, granulo, run_paths, sizing_mode)

    accepted = sum(1 for item in models if _ellipsoid_is_valid(item))
    return PipelineResult(
        params=params,
        run_paths=run_paths,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        colorized_cloud_paths=colorized_cloud_paths,
        labels_count=len(final_stacks),
        ellipsoid_count=len(models),
        accepted_ellipsoid_count=accepted,
        sizing_mode=sizing_mode,
        fit_strategy=fit_strategy,
        orientation_mode=orientation_mode,
        preprocess_mode=preprocess_mode,
        max_ellipsoid_cloud_fraction=max_ellipsoid_cloud_fraction,
        workers=effective_workers,
        ground_mode=ground_mode,
    )
