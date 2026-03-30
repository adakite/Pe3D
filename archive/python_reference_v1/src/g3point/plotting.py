from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "g3point-mpl-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


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


def plot_elevation(points: np.ndarray, stem: Path) -> None:
    sampled, _ = _sample(points)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], c=sampled[:, 2], s=1, cmap="viridis")
    fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05, label="Elevation")
    ax.set_axis_off()
    ax.set_title("Elevation")
    _save(fig, stem)


def plot_labels(points: np.ndarray, labels: np.ndarray, stem: Path) -> None:
    valid = labels > 0
    sampled_points, sampled_labels = _sample(points[valid], labels[valid], max_points=35_000)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], c=sampled_labels, s=1, cmap="tab20")
    ax.set_axis_off()
    ax.set_title(stem.name.replace("_", " "))
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
    ax.set_axis_off()
    ax.set_title("Fitted cuboids")
    _save(fig, stem)


def plot_ellipsoids(points: np.ndarray, ellipsoids: list[dict[str, np.ndarray]], stem: Path) -> None:
    sampled, _ = _sample(points)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], c="k", s=0.8, alpha=0.35)
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
    for ellipsoid in [item for item in ellipsoids if item["fitok"] and item["Aqualityok"]][:25]:
        local = base * ellipsoid["r"]
        world = local @ ellipsoid["R"].T + ellipsoid["c"]
        x = world[:, 0].reshape(v.shape[0], u.shape[0])
        y = world[:, 1].reshape(v.shape[0], u.shape[0])
        z = world[:, 2].reshape(v.shape[0], u.shape[0])
        ax.plot_wireframe(x, y, z, color="tab:blue", linewidth=0.3, alpha=0.5)
    ax.set_axis_off()
    ax.set_title("Fitted ellipsoids")
    _save(fig, stem)


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
    ax.set_title("surface cover")
    _save(fig, figure_dir / "Acover_distribution", pdf=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(np.degrees(granulo["angle_Mview"]), bins=20, color="c")
    axes[0].set_xlim(0.0, 180.0)
    axes[0].set_title("azimuth angle")
    axes[1].hist(np.degrees(granulo["angle_Xview"]), bins=20, color="m")
    axes[1].set_xlim(0.0, 180.0)
    axes[1].set_title("dip angle")
    _save(fig, figure_dir / "grain-orientation_distribution", pdf=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    radius = granulo["radius"]
    axes[0, 0].quiver(
        np.zeros_like(granulo["u_Mview"]),
        np.zeros_like(granulo["v_Mview"]),
        granulo["u_Mview"],
        granulo["v_Mview"],
        radius,
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap="viridis",
    )
    axes[0, 0].set_title("Map view")
    axes[0, 1].quiver(
        np.zeros_like(granulo["w_Xview"]),
        np.zeros_like(granulo["v_Xview"]),
        granulo["w_Xview"],
        granulo["v_Xview"],
        radius,
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap="viridis",
    )
    axes[0, 1].set_title("X-view")
    axes[1, 0].hist(np.degrees(granulo["angle_Mview"]), bins=20, color="tab:blue")
    axes[1, 1].hist(np.degrees(granulo["angle_Xview"]), bins=20, color="tab:orange")
    _save(fig, figure_dir / "grain_orientation")
