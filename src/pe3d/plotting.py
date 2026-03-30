from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "pe3d-mpl-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

MATLAB_VIEW_AZIM = -37.5
MATLAB_VIEW_ELEV = 30.0


def _save(fig: plt.Figure, stem: Path, pdf: bool = False) -> None:
    fig.savefig(stem.with_suffix(".jpg"), dpi=220, bbox_inches="tight")
    if pdf:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _sample(points: np.ndarray, values: np.ndarray | None = None, max_points: int = 25_000) -> tuple[np.ndarray, np.ndarray | None]:
    if points.shape[0] <= max_points:
        return points, values
    step = max(points.shape[0] // max_points, 1)
    sampled_points = points[::step]
    sampled_values = values[::step] if values is not None else None
    return sampled_points, sampled_values


def _style_3d_axes(ax: plt.Axes, points: np.ndarray, *, title: str | None = None) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    center = (mins + maxs) / 2.0
    radius = 0.5 * np.max(spans)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 0.35))
    ax.view_init(elev=MATLAB_VIEW_ELEV, azim=MATLAB_VIEW_AZIM)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)


def _accepted_ellipsoids(ellipsoids: list[dict[str, np.ndarray]]) -> tuple[list[int], list[dict[str, np.ndarray]]]:
    accepted_indices = [
        idx for idx, item in enumerate(ellipsoids) if item["fitok"] and item["Aqualityok"] and item.get("sizecapok", True)
    ]
    accepted = [ellipsoids[idx] for idx in accepted_indices]
    accepted.sort(key=lambda item: float(np.prod(item["r"])), reverse=True)
    index_lookup = {id(item): idx for idx, item in zip(accepted_indices, [ellipsoids[idx] for idx in accepted_indices], strict=False)}
    sorted_indices = [index_lookup[id(item)] for item in accepted]
    return sorted_indices, accepted


def plot_elevation(points: np.ndarray, stem: Path) -> None:
    sampled, _ = _sample(points)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], c=sampled[:, 2], s=1, cmap="viridis")
    fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05, label="Elevation")
    _style_3d_axes(ax, sampled, title="Elevation")
    _save(fig, stem)


def plot_labels(points: np.ndarray, labels: np.ndarray, stem: Path) -> None:
    valid = labels > 0
    sampled_points, sampled_labels = _sample(points[valid], labels[valid], max_points=35_000)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c=sampled_labels, s=1, cmap="tab20")
    _style_3d_axes(ax, sampled_points, title=stem.name.replace("_", " "))
    _save(fig, stem)


def _draw_box(ax: plt.Axes, corners: np.ndarray) -> None:
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    for a_idx, b_idx in edges:
        segment = corners[[a_idx, b_idx]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="tab:orange", linewidth=0.8)


def plot_cuboids(points: np.ndarray, cuboids: list[dict[str, np.ndarray]], stem: Path) -> None:
    sampled, _ = _sample(points)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], c="k", s=0.8, alpha=0.35)
    for cuboid in sorted(cuboids, key=lambda item: float(np.prod(item["extents"])), reverse=True)[:30]:
        _draw_box(ax, cuboid["corners"])
    _style_3d_axes(ax, sampled, title="Fitted cuboids")
    _save(fig, stem)


def plot_ellipsoids(points: np.ndarray, ellipsoids: list[dict[str, np.ndarray]], stem: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0.0, 2.0 * math.pi, 24)
    v = np.linspace(0.0, math.pi, 12)
    uu, vv = np.meshgrid(u, v)
    base = np.column_stack(
        [
            np.cos(uu).ravel() * np.sin(vv).ravel(),
            np.sin(uu).ravel() * np.sin(vv).ravel(),
            np.cos(vv).ravel(),
        ]
    )
    _, accepted = _accepted_ellipsoids(ellipsoids)
    colors = plt.cm.hsv(np.linspace(0.0, 1.0, max(len(accepted), 1), endpoint=False))
    for ellipsoid, color in zip(accepted, colors, strict=False):
        local = base * ellipsoid["r"]
        world = local @ ellipsoid["R"].T + ellipsoid["c"]
        x = world[:, 0].reshape(v.shape[0], u.shape[0])
        y = world[:, 1].reshape(v.shape[0], u.shape[0])
        z = world[:, 2].reshape(v.shape[0], u.shape[0])
        ax.plot_wireframe(x, y, z, color=color, linewidth=0.25, alpha=0.45)
    _style_3d_axes(ax, points, title="Fitted ellipsoids")
    _save(fig, stem)


def _plot_ellipsoids_colored(
    points: np.ndarray,
    accepted: list[dict[str, np.ndarray]],
    values: np.ndarray,
    stem: Path,
    title: str,
    colorbar_label: str,
    cmap_name: str,
) -> None:
    if not accepted:
        return

    values = np.asarray(values, dtype=np.float64)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0.0, 2.0 * math.pi, 24)
    v = np.linspace(0.0, math.pi, 12)
    uu, vv = np.meshgrid(u, v)
    base = np.column_stack(
        [
            np.cos(uu).ravel() * np.sin(vv).ravel(),
            np.sin(uu).ravel() * np.sin(vv).ravel(),
            np.cos(vv).ravel(),
        ]
    )
    for ellipsoid, value in zip(accepted, values, strict=False):
        local = base * ellipsoid["r"]
        world = local @ ellipsoid["R"].T + ellipsoid["c"]
        x = world[:, 0].reshape(v.shape[0], u.shape[0])
        y = world[:, 1].reshape(v.shape[0], u.shape[0])
        z = world[:, 2].reshape(v.shape[0], u.shape[0])
        ax.plot_wireframe(x, y, z, color=cmap(norm(float(value))), linewidth=0.25, alpha=0.45)

    _style_3d_axes(ax, points, title=title)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(values)
    fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.03, label=colorbar_label)
    _save(fig, stem)


def plot_ellipsoid_variants(
    points: np.ndarray,
    ellipsoids: list[dict[str, np.ndarray]],
    granulo: dict[str, np.ndarray],
    figure_dir: Path,
) -> None:
    valid_indices = np.asarray(granulo.get("valid_indices", np.empty(0, dtype=np.int64)), dtype=np.int64)
    if valid_indices.size == 0:
        return

    accepted_indices, accepted = _accepted_ellipsoids(ellipsoids)
    position_lookup = {int(idx): pos for pos, idx in enumerate(valid_indices.tolist())}

    volume = np.array([float(ellipsoids[idx]["V"]) for idx in valid_indices], dtype=np.float64)
    azimuth = np.degrees(np.asarray(granulo["angle_Mview"], dtype=np.float64))
    diameters = np.asarray(granulo["diameter"], dtype=np.float64)
    elongation = 1.0 - (diameters[2] / diameters[0])
    volume_sorted = np.array([volume[position_lookup[int(idx)]] for idx in accepted_indices], dtype=np.float64)
    azimuth_sorted = np.array([azimuth[position_lookup[int(idx)]] for idx in accepted_indices], dtype=np.float64)
    elongation_sorted = np.array([elongation[position_lookup[int(idx)]] for idx in accepted_indices], dtype=np.float64)

    _plot_ellipsoids_colored(
        points,
        accepted,
        np.log10(np.maximum(volume_sorted, 1e-12)),
        figure_dir / "fitted_ellipsoids_by_volume",
        "Fitted ellipsoids by volume",
        "log10(volume m^3)",
        "viridis",
    )
    _plot_ellipsoids_colored(
        points,
        accepted,
        azimuth_sorted,
        figure_dir / "fitted_ellipsoids_by_azimuth",
        "Fitted ellipsoids by azimuth",
        "azimuth (deg)",
        "twilight",
    )
    _plot_ellipsoids_colored(
        points,
        accepted,
        elongation_sorted,
        figure_dir / "fitted_ellipsoids_by_elongation",
        "Fitted ellipsoids by elongation",
        "elongation = 1 - c/a",
        "plasma",
    )


def plot_granulo(granulo: dict[str, np.ndarray], figure_dir: Path) -> None:
    if granulo.get("diameter", np.empty((3, 0))).size == 0:
        return

    diameters = granulo["diameter"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["r", "b", "g"]
    titles = ["a axis", "b axis", "c axis"]
    for idx, ax in enumerate(axes):
        ax.hist(diameters[idx], bins=granulo["diameter_edges_log"], color=colors[idx])
        ax.set_xscale("log")
        ax.set_title(titles[idx])
        ax.set_xlabel("diameter (m)")
        ax.set_ylabel("N")
    _save(fig, figure_dir / "grain-size_distribution_log", pdf=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for idx, ax in enumerate(axes):
        ax.hist(diameters[idx], bins=granulo["diameter_edges_lin"], color=colors[idx])
        ax.set_title(titles[idx])
        ax.set_xlabel("diameter (m)")
        ax.set_ylabel("N")
    _save(fig, figure_dir / "grain-size_distribution_lin", pdf=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(granulo["vol"], bins=granulo["vol_edges_log"], color="k")
    axes[0].set_xscale("log")
    axes[0].set_title("volume")
    axes[1].hist(granulo["area"], bins=granulo["area_edges_log"], color="k")
    axes[1].set_xscale("log")
    axes[1].set_title("area")
    _save(fig, figure_dir / "volume-area_distribution_log", pdf=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(diameters[2] / diameters[0], bins=granulo["nbin"], color="k")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_title("c/a")
    axes[1].hist(diameters[1] / diameters[0], bins=granulo["nbin"], color="k")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_title("b/a")
    _save(fig, figure_dir / "axis-ratio_distribution", pdf=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(granulo["Acover"], bins=granulo["nbin"], color="k")
    ax.set_xlim(0.0, 100.0)
    ax.set_title(str(granulo.get("Acover_label", "surface cover (%)")))
    _save(fig, figure_dir / "Acover_distribution", pdf=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    angle_map_deg = np.degrees(granulo["angle_Mview"])
    angle_x_deg = np.degrees(granulo["angle_Xview"])
    angle_bins_deg = np.linspace(0.0, 180.0, 21)
    axes[0].hist(angle_map_deg, bins=angle_bins_deg, color="c")
    axes[0].set_xlim(0.0, 180.0)
    axes[0].set_title("azimuth angle")
    axes[1].hist(angle_x_deg, bins=angle_bins_deg, color="m")
    axes[1].set_xlim(0.0, 180.0)
    axes[1].set_title("dip angle")
    _save(fig, figure_dir / "grain-orientation_distribution", pdf=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={"projection": "polar"})
    radius = granulo["radius"]
    radius_max = max(float(np.max(radius)), 1e-6)
    angle_bins_rad = np.linspace(0.0, math.pi, 21)
    map_plot_angles = (math.pi - np.asarray(granulo["angle_Mview"], dtype=np.float64)) % math.pi
    x_plot_angles = np.asarray(granulo["angle_Xview"], dtype=np.float64)
    panels = [
        (axes[0, 0], map_plot_angles, "Map view", "viridis"),
        (axes[0, 1], x_plot_angles, "X-view", "plasma"),
    ]
    for ax, angles, title, cmap_name in panels:
        cmap = plt.get_cmap(cmap_name)
        for angle, r in zip(angles, radius, strict=False):
            ax.plot([angle, angle], [0.0, r], color=cmap(min(r / radius_max, 1.0)), linewidth=0.9, alpha=0.85)
        ax.set_thetamin(0.0)
        ax.set_thetamax(180.0)
        ax.set_ylim(0.0, radius_max)
        ax.set_title(title)

    rose_panels = [
        (axes[1, 0], map_plot_angles, "Map rose", "tab:blue"),
        (axes[1, 1], x_plot_angles, "X-view rose", "tab:orange"),
    ]
    for ax, angles, title, color in rose_panels:
        ax.hist(angles, bins=angle_bins_rad, color=color)
        ax.set_thetamin(0.0)
        ax.set_thetamax(180.0)
        ax.set_title(title)
    _save(fig, figure_dir / "grain_orientation")
