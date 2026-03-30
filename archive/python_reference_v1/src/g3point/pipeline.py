from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree, distance

from .config import Parameters, RunPaths, create_run_paths, load_parameters
from .io import PointCloud, export_grains, export_granulo, load_pointcloud
from .math3d import (
    EPS,
    angles_to_reference,
    detrend_quadratic,
    estimate_normals,
    fibonacci_sphere,
    fit_plane,
    normalize,
    orient_vectors,
    rotation_matrix_between,
)
from .plotting import plot_cuboids, plot_elevation, plot_ellipsoids, plot_granulo, plot_labels
from .validation import ValidationSummary, validate_against_reference


RADFACTOR_CORRECTION = 5.0 / 6.0


@dataclass(slots=True)
class PipelineResult:
    params: Parameters
    run_paths: RunPaths
    csv_path: Path | None
    xlsx_path: Path | None
    labels_count: int
    ellipsoid_count: int
    accepted_ellipsoid_count: int
    validation: ValidationSummary | None


def statistical_denoise(points: np.ndarray, k: int) -> np.ndarray:
    if points.shape[0] <= k + 1:
        return np.ones(points.shape[0], dtype=bool)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(k + 1, points.shape[0]))
    neighbour_mean = distances[:, 1:].mean(axis=1)
    threshold = neighbour_mean.mean() + 2.0 * neighbour_mean.std()
    return neighbour_mean <= threshold


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        return points
    voxels = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((counts.size, 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def multiscale_plan_distance(points: np.ndarray, params: Parameters, rng: np.random.Generator) -> np.ndarray:
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
        neighbourhoods = tree.query_ball_point(points[center_indices], radius)
        for neighbour_idx in neighbourhoods:
            if len(neighbour_idx) < 6:
                continue
            neighbour_idx = np.asarray(neighbour_idx, dtype=np.int64)
            _, _, _, _, distabs = fit_plane(points[neighbour_idx])
            distances_sum[neighbour_idx] += distabs / max(radius, EPS)
            counts[neighbour_idx] += 1.0

    counts[counts == 0.0] = 1.0
    return distances_sum / counts


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
    # This correction keeps the Python merge stage close to the bundled MATLAB
    # reference on Otira despite slightly different neighbour geometry.
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


def _surface_area_ellipsoid(radii: np.ndarray) -> float:
    power = 1.6075
    a, b, c = radii
    return float(4.0 * math.pi * (((a * b) ** power + (a * c) ** power + (b * c) ** power) / 3.0) ** (1.0 / power))


def _fit_ellipsoid(points: np.ndarray, surface_sum: float, params: Parameters, rng: np.random.Generator) -> dict[str, object]:
    model: dict[str, object] = {"fitok": False, "Aqualityok": False}
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
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    rotation = rotation[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0

    local = (points - centre) @ rotation
    scale = np.sqrt(np.sum((local / radii) ** 2, axis=1))
    scale = np.maximum(scale, EPS)
    nearest_local = local / scale[:, None]
    distances = np.linalg.norm(local - nearest_local, axis=1)
    radial_residual = scale - 1.0
    ss_res = float(np.sum(radial_residual * radial_residual))
    ss_tot = float(np.sum((scale - scale.mean()) ** 2)) + EPS

    volume = float((4.0 / 3.0) * math.pi * np.prod(radii))
    area = _surface_area_ellipsoid(radii)
    sphere = fibonacci_sphere(200)
    sample_points = (sphere * radii) @ rotation.T + centre
    sample_tree = cKDTree(sample_points)
    _, matched = sample_tree.query(points, k=1)
    acover = 100.0 * np.unique(matched).size / max(sample_points.shape[0], 1)

    model.update(
        {
            "fitok": True,
            "Aqualityok": acover > params.aquality_thresh,
            "c": centre,
            "r": radii,
            "R": rotation,
            "axis1": rotation[:, 0],
            "axis2": rotation[:, 1],
            "axis3": rotation[:, 2],
            "d": distances,
            "r2": 1.0 - ss_res / ss_tot,
            "V": volume,
            "A": area,
            "Aratio": surface_sum / max(area, EPS),
            "Acover": acover,
        }
    )
    return model


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
    }


def grain_size_distribution(ellipsoids: list[dict[str, object]]) -> dict[str, np.ndarray | int]:
    valid_indices = np.array([idx for idx, item in enumerate(ellipsoids) if item.get("fitok") and item.get("Aqualityok")], dtype=np.int64)
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
    }
    return granulo


def ellipsoid_orientation(points_rot: np.ndarray, ellipsoids: list[dict[str, object]], granulo: dict[str, np.ndarray | int]) -> dict[str, np.ndarray | int]:
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
        axis = np.asarray(ellipsoid["axis1"], dtype=np.float64).copy()
        if sensor_map @ axis < 0.0:
            axis *= -1.0
        u_map.append(axis[0])
        v_map.append(axis[1])
        w_map.append(axis[2])
        angle_map.append(math.atan2(axis[1], axis[0]) + math.pi / 2.0)

        axis_x = np.asarray(ellipsoid["axis1"], dtype=np.float64).copy()
        if sensor_x @ axis_x < 0.0:
            axis_x *= -1.0
        u_x.append(axis_x[0])
        v_x.append(axis_x[1])
        w_x.append(axis_x[2])
        angle_x.append(math.atan2(axis_x[1], axis_x[2]) + math.pi / 2.0)

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
    ellipsoids: list[dict[str, object]],
    granulo: dict[str, np.ndarray | int],
    run_paths: RunPaths,
) -> None:
    plot_elevation(points_rot, run_paths.figure_dir / "elevation")
    plot_labels(points, initial_labels, run_paths.figure_dir / "labels_ini")
    plot_labels(points, clustered_labels, run_paths.figure_dir / "labels_cluster")
    if cleaned_labels is not None:
        plot_labels(points, cleaned_labels, run_paths.figure_dir / "labels_clean")
    plot_cuboids(points, cuboids, run_paths.figure_dir / "fitted_cuboids")
    plot_ellipsoids(points, ellipsoids, run_paths.figure_dir / "fitted_ellipsoids")
    plot_granulo(granulo, run_paths.figure_dir)


def run_pipeline(
    pointcloud_path: str | Path,
    *,
    param_csv: str | Path | None = None,
    output_root: str | Path = "python_outputs",
    reference_root: str | Path | None = None,
    validate: bool = True,
    save_plots: bool | None = None,
    save_grains: bool | None = None,
    seed: int = 0,
) -> PipelineResult:
    rng = np.random.default_rng(seed)
    params = load_parameters(pointcloud_path, param_csv=param_csv)
    run_paths = create_run_paths(output_root, params.pointcloud_name)
    cloud = load_pointcloud(params.pointcloud_path)
    params.iscolor = cloud.iscolor

    points = cloud.points
    if params.denoise:
        points = points[statistical_denoise(points, params.nnptcloud)]
    if params.decimate:
        points = voxel_downsample(points, params.res)
    if params.minima:
        roughness = multiscale_plan_distance(points, params, rng)
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
    distances, neighbour_idx = tree.query(points, k=min(params.nnptcloud + 1, points.shape[0]))
    if neighbour_idx.ndim == 1:
        neighbour_idx = neighbour_idx[:, None]
        distances = distances[:, None]
    if neighbour_idx.shape[1] > 1:
        neighbour_idx = neighbour_idx[:, 1:]
        distances = distances[:, 1:]
    surface = math.pi * np.square(np.maximum(np.min(distances, axis=1), 0.0))

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

    cleaned_labels = None
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

    cuboids = [fit_pca_cuboid(points[stack]) for stack in final_stacks]
    ellipsoids = [_fit_ellipsoid(points[stack], float(surface[stack].sum()), params, rng) for stack in final_stacks]
    granulo = ellipsoid_orientation(points_rot, ellipsoids, grain_size_distribution(ellipsoids))

    csv_path = None
    xlsx_path = None
    if save_grains if save_grains is not None else params.savegrain:
        export_grains(points, final_labels, run_paths, params.pointcloud_name)
    if params.savegranulo:
        csv_path, xlsx_path = export_granulo(run_paths, params.pointcloud_name, granulo)
    if save_plots if save_plots is not None else params.saveplot:
        _plot_outputs(points, points_rot, initial_labels, clustered_labels, cleaned_labels, cuboids, ellipsoids, granulo, run_paths)

    validation_summary = None
    if validate and csv_path and reference_root is not None:
        reference_root = Path(reference_root)
        stem = Path(params.pointcloud_name).stem
        reference_csv = reference_root / "Excel" / f"{stem}_n1" / f"{params.pointcloud_name}_granulo.csv"
        reference_grains = reference_root / "Grain" / f"{stem}_n1"
        if reference_csv.exists() and reference_grains.exists():
            validation_summary = validate_against_reference(csv_path, reference_csv, run_paths.grain_dir, reference_grains)

    accepted = sum(1 for item in ellipsoids if item.get("fitok") and item.get("Aqualityok"))
    return PipelineResult(
        params=params,
        run_paths=run_paths,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        labels_count=len(final_stacks),
        ellipsoid_count=len(ellipsoids),
        accepted_ellipsoid_count=accepted,
        validation=validation_summary,
    )
