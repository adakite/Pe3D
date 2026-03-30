from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from inspect import signature
from pathlib import Path

import numpy as np

cache_root = Path(tempfile.gettempdir()) / "pe3d-mpl-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

if matplotlib.get_backend().lower().endswith("agg"):
    for backend in ("TkAgg", "QtAgg", "macosx"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, PolygonSelector, RadioButtons

from .io import load_pointcloud, write_pointcloud
from .math3d import detrend_quadratic, fit_plane, normalize, orient_vectors, rotation_matrix_between


LABEL_UNKNOWN = 0
LABEL_BLOCK = 1
LABEL_NON_BLOCK = -1
LABEL_MODES = {
    "block": LABEL_BLOCK,
    "non-block": LABEL_NON_BLOCK,
}
PLY_LABEL_RGB = {
    LABEL_UNKNOWN: np.array([120, 120, 120], dtype=np.uint8),
    LABEL_BLOCK: np.array([222, 92, 48], dtype=np.uint8),
    LABEL_NON_BLOCK: np.array([44, 123, 182], dtype=np.uint8),
}
DISPLAY_LABEL_RGBA = {
    LABEL_BLOCK: np.array([0.93, 0.36, 0.18, 0.95], dtype=np.float64),
    LABEL_NON_BLOCK: np.array([0.17, 0.48, 0.72, 0.95], dtype=np.float64),
}


@dataclass(slots=True)
class ManualLabelPaths:
    session_dir: Path
    labels_path: Path
    metadata_path: Path
    polygons_path: Path
    preview_ply_path: Path


@dataclass(slots=True)
class PreparedLabelData:
    pointcloud_path: Path
    raw_points: np.ndarray
    points_view: np.ndarray
    sample_indices: np.ndarray
    rotation: np.ndarray
    origin: np.ndarray
    plane_coefficients: np.ndarray
    plane_normal: np.ndarray
    rotate_horizontal: bool
    detrend: bool


@dataclass(slots=True)
class ManualLabelResult:
    pointcloud_path: Path
    session_dir: Path
    labels_path: Path
    metadata_path: Path
    polygons_path: Path
    preview_ply_path: Path
    block_count: int
    non_block_count: int
    unknown_count: int
    instance_count: int = 0


def create_manual_label_paths(
    output_root: str | Path,
    pointcloud_name: str,
    *,
    create_dir: bool = True,
    session_prefix: str = "manual_labels",
) -> ManualLabelPaths:
    output_root = Path(output_root)
    stem = Path(pointcloud_name).stem
    session_dir = output_root / stem
    if create_dir:
        session_dir.mkdir(parents=True, exist_ok=True)
    if session_prefix == "manual_labels":
        polygons_path = session_dir / f"{stem}_manual_polygons.json"
    else:
        polygons_path = session_dir / f"{stem}_{session_prefix}_polygons.json"
    return ManualLabelPaths(
        session_dir=session_dir,
        labels_path=session_dir / f"{stem}_{session_prefix}.npz",
        metadata_path=session_dir / f"{stem}_{session_prefix}.json",
        polygons_path=polygons_path,
        preview_ply_path=session_dir / f"{stem}_{session_prefix}.ply",
    )


def clear_manual_label_session(
    pointcloud_path: str | Path,
    *,
    output_root: str | Path = "pe3d_labels",
    session_prefix: str = "manual_labels",
) -> ManualLabelPaths:
    paths = create_manual_label_paths(output_root, Path(pointcloud_path).name, create_dir=False, session_prefix=session_prefix)
    for path in (paths.labels_path, paths.metadata_path, paths.polygons_path, paths.preview_ply_path):
        if path.exists():
            path.unlink()
    if paths.session_dir.exists():
        try:
            paths.session_dir.rmdir()
        except OSError:
            pass
    return paths


def _prepare_view_points(
    points: np.ndarray,
    *,
    rotate_horizontal: bool,
    detrend: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points_view = points.copy()
    origin = points.mean(axis=0) if points.size else np.zeros(3, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    plane_coefficients = np.zeros(3, dtype=np.float64)
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if not points.size:
        return points_view, rotation, origin, plane_coefficients, plane_normal

    a, b, c, _, _ = fit_plane(points)
    plane_coefficients = np.array([a, b, c], dtype=np.float64)
    plane_normal = normalize(np.array([[-a, -b, 1.0]], dtype=np.float64))[0]
    plane_normal = orient_vectors(np.zeros((1, 3)), plane_normal.reshape(1, 3), np.array([0.0, 0.0, 1e32]))[0]

    if rotate_horizontal:
        rotation = rotation_matrix_between(plane_normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        points_view = (points - origin) @ rotation.T + origin
    if detrend:
        points_view, _ = detrend_quadratic(points_view)

    return points_view, rotation, origin, plane_coefficients, plane_normal


def _choose_sample_indices(
    count: int,
    max_display_points: int,
    rng: np.random.Generator,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    if count <= max_display_points:
        return np.arange(count, dtype=np.int64)
    if labels is None:
        return np.sort(rng.choice(count, size=max_display_points, replace=False).astype(np.int64))

    labelled = np.flatnonzero(labels != LABEL_UNKNOWN)
    if labelled.size >= max_display_points:
        return np.sort(rng.choice(labelled, size=max_display_points, replace=False).astype(np.int64))

    remaining = max_display_points - labelled.size
    unlabeled = np.flatnonzero(labels == LABEL_UNKNOWN)
    fill = rng.choice(unlabeled, size=min(remaining, unlabeled.size), replace=False).astype(np.int64)
    return np.sort(np.unique(np.concatenate([labelled.astype(np.int64, copy=False), fill])))


def _default_polygon_state() -> dict[str, object]:
    return {
        "zones": {
            "block": None,
            "non-block": None,
        },
        "mode_order": ["block", "non-block"],
    }


def _normalize_polygon_state(payload: object) -> dict[str, object]:
    state = _default_polygon_state()
    zones = state["zones"]

    if isinstance(payload, dict):
        if isinstance(payload.get("zones"), dict):
            source = payload["zones"]
        else:
            source = payload
        for mode in ("block", "non-block"):
            vertices = source.get(mode)
            if isinstance(vertices, list) and len(vertices) >= 3:
                zones[mode] = [[float(x), float(y)] for x, y in vertices]
        mode_order = payload.get("mode_order")
        if isinstance(mode_order, list):
            filtered = [mode for mode in mode_order if mode in ("block", "non-block")]
            if filtered:
                state["mode_order"] = filtered
        return state

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            mode = item.get("mode")
            vertices = item.get("vertices_xy")
            if mode in ("block", "non-block") and isinstance(vertices, list) and len(vertices) >= 3:
                zones[mode] = [[float(x), float(y)] for x, y in vertices]
                state["mode_order"] = [m for m in state["mode_order"] if m != mode] + [mode]
    return state


def _load_existing_labels(paths: ManualLabelPaths, point_count: int) -> tuple[np.ndarray, dict[str, object]]:
    labels = np.zeros(point_count, dtype=np.int8)
    polygon_state = _default_polygon_state()

    if paths.labels_path.exists():
        with np.load(paths.labels_path) as data:
            loaded = np.asarray(data["labels"], dtype=np.int8)
        if loaded.shape != (point_count,):
            raise RuntimeError(
                f"Existing labels in {paths.labels_path} have {loaded.shape[0]} entries, "
                f"but the point cloud contains {point_count} points."
            )
        labels = loaded

    if paths.polygons_path.exists():
        polygon_state = _normalize_polygon_state(json.loads(paths.polygons_path.read_text(encoding="utf-8")))

    return labels, polygon_state


def _points_in_polygon(points_xy: np.ndarray, vertices: list[list[float]] | None) -> np.ndarray:
    if vertices is None or len(vertices) < 3:
        return np.zeros(points_xy.shape[0], dtype=bool)
    polygon = np.asarray(vertices, dtype=np.float64)
    xmin, ymin = np.min(polygon, axis=0)
    xmax, ymax = np.max(polygon, axis=0)
    bbox_mask = (
        (points_xy[:, 0] >= xmin)
        & (points_xy[:, 0] <= xmax)
        & (points_xy[:, 1] >= ymin)
        & (points_xy[:, 1] <= ymax)
    )
    candidate_idx = np.flatnonzero(bbox_mask)
    if candidate_idx.size == 0:
        return bbox_mask
    inside = np.zeros(points_xy.shape[0], dtype=bool)
    inside[candidate_idx] = MplPath(polygon).contains_points(points_xy[candidate_idx])
    return inside


def _labels_from_polygon_state(points_view: np.ndarray, polygon_state: dict[str, object]) -> np.ndarray:
    labels = np.zeros(points_view.shape[0], dtype=np.int8)
    zones = polygon_state.get("zones", {})
    mode_order = [mode for mode in polygon_state.get("mode_order", ["block", "non-block"]) if mode in ("block", "non-block")]
    if not mode_order:
        mode_order = ["block", "non-block"]
    for mode in mode_order:
        mask = _points_in_polygon(points_view[:, :2], zones.get(mode))
        labels[mask] = LABEL_MODES[mode]
    return labels


def prepare_manual_label_data(
    pointcloud_path: str | Path,
    *,
    output_root: str | Path = "pe3d_labels",
    max_display_points: int = 120_000,
    seed: int = 0,
    rotate_horizontal: bool = True,
    detrend: bool = False,
) -> tuple[PreparedLabelData, ManualLabelPaths, np.ndarray, dict[str, object]]:
    cloud = load_pointcloud(pointcloud_path)
    paths = create_manual_label_paths(output_root, Path(pointcloud_path).name)
    labels, polygon_state = _load_existing_labels(paths, cloud.count)

    rng = np.random.default_rng(seed)
    points_view, rotation, origin, plane_coefficients, plane_normal = _prepare_view_points(
        cloud.points,
        rotate_horizontal=rotate_horizontal,
        detrend=detrend,
    )
    if polygon_state["zones"]["block"] is not None or polygon_state["zones"]["non-block"] is not None:
        labels = _labels_from_polygon_state(points_view, polygon_state)
    sample_indices = _choose_sample_indices(cloud.count, max_display_points, rng, labels)

    prepared = PreparedLabelData(
        pointcloud_path=Path(pointcloud_path),
        raw_points=cloud.points,
        points_view=points_view,
        sample_indices=sample_indices,
        rotation=rotation,
        origin=origin,
        plane_coefficients=plane_coefficients,
        plane_normal=plane_normal,
        rotate_horizontal=rotate_horizontal,
        detrend=detrend,
    )
    return prepared, paths, labels, polygon_state


def _label_colors_for_ply(labels: np.ndarray) -> np.ndarray:
    colors = np.empty((labels.shape[0], 3), dtype=np.uint8)
    colors[:] = PLY_LABEL_RGB[LABEL_UNKNOWN]
    for label_code, rgb in PLY_LABEL_RGB.items():
        colors[labels == label_code] = rgb
    return colors


def _metadata_from_labels(
    prepared: PreparedLabelData,
    labels: np.ndarray,
    paths: ManualLabelPaths,
    polygon_state: dict[str, object],
) -> dict[str, object]:
    block_count = int(np.count_nonzero(labels == LABEL_BLOCK))
    non_block_count = int(np.count_nonzero(labels == LABEL_NON_BLOCK))
    unknown_count = int(np.count_nonzero(labels == LABEL_UNKNOWN))
    zones = polygon_state.get("zones", {})
    polygon_count = int(sum(zones.get(mode) is not None for mode in ("block", "non-block")))
    return {
        "pointcloud": str(prepared.pointcloud_path),
        "point_count": int(labels.size),
        "rotate_horizontal": bool(prepared.rotate_horizontal),
        "detrend": bool(prepared.detrend),
        "plane_coefficients": prepared.plane_coefficients.tolist(),
        "plane_normal": prepared.plane_normal.tolist(),
        "rotation_matrix": prepared.rotation.tolist(),
        "origin": prepared.origin.tolist(),
        "display_point_count": int(prepared.sample_indices.size),
        "block_count": block_count,
        "non_block_count": non_block_count,
        "unknown_count": unknown_count,
        "polygon_count": polygon_count,
        "labels_path": str(paths.labels_path),
        "metadata_path": str(paths.metadata_path),
        "polygons_path": str(paths.polygons_path),
        "preview_ply_path": str(paths.preview_ply_path),
        "polygon_modes": {
            "block": zones.get("block") is not None,
            "non-block": zones.get("non-block") is not None,
        },
        "label_codes": {
            "unknown": LABEL_UNKNOWN,
            "block": LABEL_BLOCK,
            "non-block": LABEL_NON_BLOCK,
        },
    }


def save_manual_label_session(
    prepared: PreparedLabelData,
    labels: np.ndarray,
    paths: ManualLabelPaths,
    polygon_state: dict[str, object],
) -> ManualLabelResult:
    labels = np.asarray(labels, dtype=np.int8)
    np.savez_compressed(
        paths.labels_path,
        labels=labels,
        rotation=prepared.rotation,
        origin=prepared.origin,
        plane_coefficients=prepared.plane_coefficients,
        plane_normal=prepared.plane_normal,
    )
    metadata = _metadata_from_labels(prepared, labels, paths, polygon_state)
    paths.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths.polygons_path.write_text(json.dumps(polygon_state, indent=2), encoding="utf-8")
    write_pointcloud(paths.preview_ply_path, prepared.raw_points, _label_colors_for_ply(labels), text=False)

    return ManualLabelResult(
        pointcloud_path=prepared.pointcloud_path,
        session_dir=paths.session_dir,
        labels_path=paths.labels_path,
        metadata_path=paths.metadata_path,
        polygons_path=paths.polygons_path,
        preview_ply_path=paths.preview_ply_path,
        block_count=int(metadata["block_count"]),
        non_block_count=int(metadata["non_block_count"]),
        unknown_count=int(metadata["unknown_count"]),
    )


def _default_instance_state() -> dict[str, object]:
    return {"instances": []}


def _normalize_instance_state(payload: object) -> dict[str, object]:
    state = _default_instance_state()
    instances: list[dict[str, object]] = []

    if isinstance(payload, dict):
        source = payload.get("instances", payload)
        if isinstance(source, list):
            iterable = source
        else:
            iterable = []
    elif isinstance(payload, list):
        iterable = payload
    else:
        iterable = []

    next_id = 1
    for item in iterable:
        if not isinstance(item, dict):
            continue
        vertices = item.get("vertices_xy", item.get("vertices"))
        if not isinstance(vertices, list) or len(vertices) < 3:
            continue
        instance_id = int(item.get("id", next_id))
        next_id = max(next_id, instance_id + 1)
        instances.append(
            {
                "id": instance_id,
                "vertices_xy": [[float(x), float(y)] for x, y in vertices],
            }
        )

    instances.sort(key=lambda item: int(item["id"]))
    state["instances"] = instances
    return state


def _labels_from_instance_state(points_view: np.ndarray, instance_state: dict[str, object]) -> np.ndarray:
    labels = np.zeros(points_view.shape[0], dtype=np.int32)
    for item in instance_state.get("instances", []):
        mask = _points_in_polygon(points_view[:, :2], item.get("vertices_xy"))
        labels[mask] = int(item["id"])
    return labels


def prepare_manual_instance_data(
    pointcloud_path: str | Path,
    *,
    output_root: str | Path = "pe3d_labels",
    max_display_points: int = 120_000,
    seed: int = 0,
    rotate_horizontal: bool = True,
    detrend: bool = False,
) -> tuple[PreparedLabelData, ManualLabelPaths, np.ndarray, dict[str, object]]:
    cloud = load_pointcloud(pointcloud_path)
    paths = create_manual_label_paths(output_root, Path(pointcloud_path).name, session_prefix="manual_instances")
    labels = np.zeros(cloud.count, dtype=np.int32)
    instance_state = _default_instance_state()

    if paths.labels_path.exists():
        with np.load(paths.labels_path) as data:
            loaded = np.asarray(data["labels"], dtype=np.int32)
        if loaded.shape != (cloud.count,):
            raise RuntimeError(
                f"Existing instance labels in {paths.labels_path} have {loaded.shape[0]} entries, "
                f"but the point cloud contains {cloud.count} points."
            )
        labels = loaded

    if paths.polygons_path.exists():
        instance_state = _normalize_instance_state(json.loads(paths.polygons_path.read_text(encoding="utf-8")))

    rng = np.random.default_rng(seed)
    points_view, rotation, origin, plane_coefficients, plane_normal = _prepare_view_points(
        cloud.points,
        rotate_horizontal=rotate_horizontal,
        detrend=detrend,
    )
    if instance_state["instances"]:
        labels = _labels_from_instance_state(points_view, instance_state)
    sample_indices = _choose_sample_indices(cloud.count, max_display_points, rng, labels)

    prepared = PreparedLabelData(
        pointcloud_path=Path(pointcloud_path),
        raw_points=cloud.points,
        points_view=points_view,
        sample_indices=sample_indices,
        rotation=rotation,
        origin=origin,
        plane_coefficients=plane_coefficients,
        plane_normal=plane_normal,
        rotate_horizontal=rotate_horizontal,
        detrend=detrend,
    )
    return prepared, paths, labels, instance_state


def _instance_label_colors(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    colors = np.empty((labels.shape[0], 3), dtype=np.uint8)
    colors[:] = PLY_LABEL_RGB[LABEL_UNKNOWN]
    positive = labels > 0
    if not np.any(positive):
        return colors
    ids = labels[positive].astype(np.float64)
    vmax = max(float(np.max(ids)), 1.0)
    rgba = cm.get_cmap("tab20")(Normalize(vmin=1.0, vmax=vmax)(ids))
    colors[positive] = np.rint(255.0 * rgba[:, :3]).astype(np.uint8)
    return colors


def _metadata_from_instances(
    prepared: PreparedLabelData,
    labels: np.ndarray,
    paths: ManualLabelPaths,
    instance_state: dict[str, object],
) -> dict[str, object]:
    labelled_count = int(np.count_nonzero(labels > 0))
    unknown_count = int(np.count_nonzero(labels <= 0))
    instance_ids = [int(item["id"]) for item in instance_state.get("instances", [])]
    return {
        "pointcloud": str(prepared.pointcloud_path),
        "point_count": int(labels.size),
        "rotate_horizontal": bool(prepared.rotate_horizontal),
        "detrend": bool(prepared.detrend),
        "plane_coefficients": prepared.plane_coefficients.tolist(),
        "plane_normal": prepared.plane_normal.tolist(),
        "rotation_matrix": prepared.rotation.tolist(),
        "origin": prepared.origin.tolist(),
        "display_point_count": int(prepared.sample_indices.size),
        "labelled_point_count": labelled_count,
        "unknown_count": unknown_count,
        "instance_count": int(len(instance_ids)),
        "instance_ids": instance_ids,
        "labels_path": str(paths.labels_path),
        "metadata_path": str(paths.metadata_path),
        "polygons_path": str(paths.polygons_path),
        "preview_ply_path": str(paths.preview_ply_path),
        "label_codes": {
            "unknown": 0,
            "instances": "positive integer ids",
        },
    }


def save_manual_instance_session(
    prepared: PreparedLabelData,
    labels: np.ndarray,
    paths: ManualLabelPaths,
    instance_state: dict[str, object],
) -> ManualLabelResult:
    labels = np.asarray(labels, dtype=np.int32)
    np.savez_compressed(
        paths.labels_path,
        labels=labels,
        rotation=prepared.rotation,
        origin=prepared.origin,
        plane_coefficients=prepared.plane_coefficients,
        plane_normal=prepared.plane_normal,
    )
    metadata = _metadata_from_instances(prepared, labels, paths, instance_state)
    paths.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths.polygons_path.write_text(json.dumps(instance_state, indent=2), encoding="utf-8")
    write_pointcloud(paths.preview_ply_path, prepared.raw_points, _instance_label_colors(labels), text=False)

    return ManualLabelResult(
        pointcloud_path=prepared.pointcloud_path,
        session_dir=paths.session_dir,
        labels_path=paths.labels_path,
        metadata_path=paths.metadata_path,
        polygons_path=paths.polygons_path,
        preview_ply_path=paths.preview_ply_path,
        block_count=int(metadata["labelled_point_count"]),
        non_block_count=0,
        unknown_count=int(metadata["unknown_count"]),
        instance_count=int(metadata["instance_count"]),
    )


class ManualLabelApp:
    def __init__(
        self,
        prepared: PreparedLabelData,
        paths: ManualLabelPaths,
        labels: np.ndarray,
        polygon_state: dict[str, object],
    ) -> None:
        self.prepared = prepared
        self.paths = paths
        self.labels = np.asarray(labels, dtype=np.int8)
        self.polygon_state = _normalize_polygon_state(polygon_state)
        self.active_mode = "block"
        self.is_dirty = False
        self.save_result: ManualLabelResult | None = None

        self.sample_points_raw = self.prepared.raw_points[self.prepared.sample_indices]
        self.sample_points_view = self.prepared.points_view[self.prepared.sample_indices]
        self.sample_base_rgba = self._height_rgba(self.sample_points_view[:, 2])

        self.fig = plt.figure(figsize=(16, 9))
        self.ax_3d = self.fig.add_axes([0.05, 0.08, 0.48, 0.84], projection="3d")
        self.ax_map = self.fig.add_axes([0.57, 0.18, 0.26, 0.70])
        self.ax_mode = self.fig.add_axes([0.86, 0.64, 0.11, 0.12])
        self.ax_save = self.fig.add_axes([0.86, 0.56, 0.11, 0.05])
        self.ax_clear = self.fig.add_axes([0.86, 0.49, 0.11, 0.05])
        self.ax_reset = self.fig.add_axes([0.86, 0.42, 0.11, 0.05])

        self.mode_buttons = RadioButtons(self.ax_mode, ["block", "non-block"], active=0)
        self.save_button = Button(self.ax_save, "Save")
        self.clear_button = Button(self.ax_clear, "Clear Mode")
        self.reset_button = Button(self.ax_reset, "Reset All")
        self.mode_buttons.on_clicked(self._on_mode_change)
        self.save_button.on_clicked(self._on_save_click)
        self.clear_button.on_clicked(self._on_clear_click)
        self.reset_button.on_clicked(self._on_reset_click)

        self.polygon_selector = self._create_polygon_selector()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.block_patch: PolygonPatch | None = None
        self.non_block_patch: PolygonPatch | None = None
        self.count_text = self.fig.text(0.86, 0.35, "", va="top", fontsize=10)
        self.status_text = self.fig.text(0.86, 0.18, "", va="top", fontsize=10)
        self.fig.text(
            0.86,
            0.94,
            "Manual labeling\nDraw one polygon for block and one for non-block.\nSelection happens in map view.\nKeys: 1 block, 2 non-block, c clear mode, r reset, s save, q close.",
            va="top",
            fontsize=10,
        )

        self.scatter_3d = self.ax_3d.scatter(
            self.sample_points_raw[:, 0],
            self.sample_points_raw[:, 1],
            self.sample_points_raw[:, 2],
            c=self._sample_rgba(),
            s=1.2,
            depthshade=False,
        )
        self.scatter_map = self.ax_map.scatter(
            self.sample_points_view[:, 0],
            self.sample_points_view[:, 1],
            c=self._sample_rgba(),
            s=1.0,
            linewidths=0.0,
            rasterized=True,
        )

        self._setup_axes()
        self._refresh_views("Loaded session.")

    def _create_polygon_selector(self) -> PolygonSelector:
        kwargs = {
            "useblit": True,
            "props": {"color": "#f28e2b", "linewidth": 1.8, "alpha": 0.9},
            "handle_props": {"marker": "o", "markersize": 4, "mfc": "white", "mec": "#f28e2b"},
        }
        supported = signature(PolygonSelector.__init__).parameters
        if "drag_from_anywhere" in supported:
            kwargs["drag_from_anywhere"] = True
        return PolygonSelector(self.ax_map, self._on_polygon_complete, **kwargs)

    def _restart_polygon_selector(self) -> None:
        if self.polygon_selector is not None:
            try:
                self.polygon_selector.clear()
            except Exception:
                pass
            try:
                self.polygon_selector.set_visible(False)
            except Exception:
                pass
            try:
                self.polygon_selector.disconnect_events()
            except Exception:
                pass
        self.polygon_selector = self._create_polygon_selector()

    def _height_rgba(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return np.empty((0, 4), dtype=np.float64)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        rgba = cm.get_cmap("terrain")(norm(values))
        rgba[:, 3] = 0.48
        return rgba

    def _sample_rgba(self) -> np.ndarray:
        rgba = self.sample_base_rgba.copy()
        sample_labels = self.labels[self.prepared.sample_indices]
        for label_code, color in DISPLAY_LABEL_RGBA.items():
            rgba[sample_labels == label_code] = color
        return rgba

    def _setup_axes(self) -> None:
        self.ax_3d.set_title("3D cloud")
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        self.ax_3d.view_init(elev=30.0, azim=-37.5)
        self._set_equal_3d(self.ax_3d, self.sample_points_raw)

        self.ax_map.set_title("Map view")
        self.ax_map.set_xlabel("Xh")
        self.ax_map.set_ylabel("Yh")
        self.ax_map.set_aspect("equal", adjustable="box")

    def _set_equal_3d(self, axis, points: np.ndarray) -> None:
        if points.size == 0:
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_zlim(0.0, 1.0)
            return
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        centre = 0.5 * (mins + maxs)
        radius = 0.5 * float(np.max(maxs - mins))
        if radius <= 0.0:
            radius = 1.0
        axis.set_xlim(centre[0] - radius, centre[0] + radius)
        axis.set_ylim(centre[1] - radius, centre[1] + radius)
        axis.set_zlim(centre[2] - radius, centre[2] + radius)

    def _recompute_labels(self) -> None:
        self.labels = _labels_from_polygon_state(self.prepared.points_view, self.polygon_state)

    def _replace_mode_polygon(self, mode: str, vertices_xy: list[list[float]] | None) -> int:
        self.polygon_state["zones"][mode] = vertices_xy
        if mode in self.polygon_state["mode_order"]:
            self.polygon_state["mode_order"].remove(mode)
        if vertices_xy is not None:
            self.polygon_state["mode_order"].append(mode)
        self._recompute_labels()
        mask = _points_in_polygon(self.prepared.points_view[:, :2], vertices_xy)
        self.is_dirty = True
        return int(np.count_nonzero(mask))

    def _set_polygon_artists(self) -> None:
        if self.block_patch is not None:
            self.block_patch.remove()
            self.block_patch = None
        if self.non_block_patch is not None:
            self.non_block_patch.remove()
            self.non_block_patch = None

        zones = self.polygon_state["zones"]
        if zones["block"] is not None:
            self.block_patch = PolygonPatch(
                np.asarray(zones["block"], dtype=np.float64),
                closed=True,
                facecolor="#de5c30",
                edgecolor="#de5c30",
                alpha=0.18,
                linewidth=1.8,
            )
            self.ax_map.add_patch(self.block_patch)
        if zones["non-block"] is not None:
            self.non_block_patch = PolygonPatch(
                np.asarray(zones["non-block"], dtype=np.float64),
                closed=True,
                facecolor="#2c7bb6",
                edgecolor="#2c7bb6",
                alpha=0.18,
                linewidth=1.8,
            )
            self.ax_map.add_patch(self.non_block_patch)

    def _refresh_views(self, status: str) -> None:
        rgba = self._sample_rgba()
        self.scatter_3d.set_color(rgba)
        self.scatter_map.set_facecolors(rgba)
        self._set_polygon_artists()

        block_count = int(np.count_nonzero(self.labels == LABEL_BLOCK))
        non_block_count = int(np.count_nonzero(self.labels == LABEL_NON_BLOCK))
        unknown_count = int(np.count_nonzero(self.labels == LABEL_UNKNOWN))
        zones = self.polygon_state["zones"]
        self.count_text.set_text(
            f"Mode: {self.active_mode}\n"
            f"Block points: {block_count}\n"
            f"Non-block points: {non_block_count}\n"
            f"Unknown points: {unknown_count}\n"
            f"Block polygon: {'set' if zones['block'] is not None else 'empty'}\n"
            f"Non-block polygon: {'set' if zones['non-block'] is not None else 'empty'}\n"
            f"Display points: {self.prepared.sample_indices.size}"
        )
        self.status_text.set_text(status)
        self.fig.canvas.draw_idle()

    def _on_mode_change(self, mode: str) -> None:
        self.active_mode = mode
        self._restart_polygon_selector()
        self._refresh_views(f"Mode changed to '{mode}'.")

    def _on_polygon_complete(self, vertices: list[tuple[float, float]]) -> None:
        vertices_xy = [[float(x), float(y)] for x, y in vertices]
        selected_count = self._replace_mode_polygon(self.active_mode, vertices_xy)
        self._restart_polygon_selector()
        self._refresh_views(f"{self.active_mode}: polygon replaced, {selected_count} points selected.")

    def _on_save_click(self, _event) -> None:
        self.save_result = save_manual_label_session(self.prepared, self.labels, self.paths, self.polygon_state)
        self.is_dirty = False
        self._refresh_views(f"Saved labels to {self.paths.session_dir}.")

    def _on_clear_click(self, _event) -> None:
        if self.polygon_state["zones"][self.active_mode] is None:
            self._refresh_views(f"No polygon stored for '{self.active_mode}'.")
            return
        self._replace_mode_polygon(self.active_mode, None)
        self._restart_polygon_selector()
        self._refresh_views(f"Cleared polygon for '{self.active_mode}'.")

    def _on_reset_click(self, _event) -> None:
        self.polygon_state = _default_polygon_state()
        self._recompute_labels()
        self.is_dirty = True
        self._restart_polygon_selector()
        self._refresh_views("Cleared both polygons.")

    def _on_key_press(self, event) -> None:
        if event.key == "1":
            self.mode_buttons.set_active(0)
        elif event.key == "2":
            self.mode_buttons.set_active(1)
        elif event.key == "c":
            self._on_clear_click(event)
        elif event.key == "r":
            self._on_reset_click(event)
        elif event.key == "s":
            self._on_save_click(event)
        elif event.key == "q":
            plt.close(self.fig)

    def _on_close(self, _event) -> None:
        if self.save_result is None or self.is_dirty:
            self.save_result = save_manual_label_session(self.prepared, self.labels, self.paths, self.polygon_state)
            self.is_dirty = False

    def run(self) -> ManualLabelResult:
        plt.show()
        if self.save_result is None:
            self.save_result = save_manual_label_session(self.prepared, self.labels, self.paths, self.polygon_state)
        return self.save_result


class ManualInstanceApp:
    def __init__(
        self,
        prepared: PreparedLabelData,
        paths: ManualLabelPaths,
        labels: np.ndarray,
        instance_state: dict[str, object],
    ) -> None:
        self.prepared = prepared
        self.paths = paths
        self.labels = np.asarray(labels, dtype=np.int32)
        self.instance_state = _normalize_instance_state(instance_state)
        self.is_dirty = False
        self.save_result: ManualLabelResult | None = None

        self.sample_points_raw = self.prepared.raw_points[self.prepared.sample_indices]
        self.sample_points_view = self.prepared.points_view[self.prepared.sample_indices]
        self.sample_base_rgba = self._height_rgba(self.sample_points_view[:, 2])

        self.fig = plt.figure(figsize=(16, 9))
        self.ax_3d = self.fig.add_axes([0.05, 0.08, 0.48, 0.84], projection="3d")
        self.ax_map = self.fig.add_axes([0.57, 0.18, 0.26, 0.70])
        self.ax_save = self.fig.add_axes([0.86, 0.56, 0.11, 0.05])
        self.ax_undo = self.fig.add_axes([0.86, 0.49, 0.11, 0.05])
        self.ax_reset = self.fig.add_axes([0.86, 0.42, 0.11, 0.05])

        self.save_button = Button(self.ax_save, "Save")
        self.undo_button = Button(self.ax_undo, "Undo Last")
        self.reset_button = Button(self.ax_reset, "Reset All")
        self.save_button.on_clicked(self._on_save_click)
        self.undo_button.on_clicked(self._on_undo_click)
        self.reset_button.on_clicked(self._on_reset_click)

        self.polygon_selector = self._create_polygon_selector()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.instance_patches: list[PolygonPatch] = []
        self.count_text = self.fig.text(0.86, 0.35, "", va="top", fontsize=10)
        self.status_text = self.fig.text(0.86, 0.18, "", va="top", fontsize=10)
        self.fig.text(
            0.86,
            0.94,
            "Instance labeling\nEach polygon creates one block id.\nSelection happens in map view.\nKeys: u undo last, r reset, s save, q close.",
            va="top",
            fontsize=10,
        )

        self.scatter_3d = self.ax_3d.scatter(
            self.sample_points_raw[:, 0],
            self.sample_points_raw[:, 1],
            self.sample_points_raw[:, 2],
            c=self._sample_rgba(),
            s=1.2,
            depthshade=False,
        )
        self.scatter_map = self.ax_map.scatter(
            self.sample_points_view[:, 0],
            self.sample_points_view[:, 1],
            c=self._sample_rgba(),
            s=1.0,
            linewidths=0.0,
            rasterized=True,
        )

        self._setup_axes()
        self._refresh_views("Loaded instance session.")

    def _create_polygon_selector(self) -> PolygonSelector:
        kwargs = {
            "useblit": True,
            "props": {"color": "#f28e2b", "linewidth": 1.8, "alpha": 0.9},
            "handle_props": {"marker": "o", "markersize": 4, "mfc": "white", "mec": "#f28e2b"},
        }
        supported = signature(PolygonSelector.__init__).parameters
        if "drag_from_anywhere" in supported:
            kwargs["drag_from_anywhere"] = True
        return PolygonSelector(self.ax_map, self._on_polygon_complete, **kwargs)

    def _restart_polygon_selector(self) -> None:
        if self.polygon_selector is not None:
            try:
                self.polygon_selector.clear()
            except Exception:
                pass
            try:
                self.polygon_selector.set_visible(False)
            except Exception:
                pass
            try:
                self.polygon_selector.disconnect_events()
            except Exception:
                pass
        self.polygon_selector = self._create_polygon_selector()

    def _height_rgba(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return np.empty((0, 4), dtype=np.float64)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        rgba = cm.get_cmap("terrain")(norm(values))
        rgba[:, 3] = 0.42
        return rgba

    def _sample_rgba(self) -> np.ndarray:
        rgba = self.sample_base_rgba.copy()
        sample_labels = self.labels[self.prepared.sample_indices]
        positive = sample_labels > 0
        if np.any(positive):
            ids = sample_labels[positive].astype(np.float64)
            vmax = max(float(np.max(self.labels)), 1.0)
            rgba[positive] = cm.get_cmap("tab20")(Normalize(vmin=1.0, vmax=vmax)(ids))
            rgba[positive, 3] = 0.95
        return rgba

    def _setup_axes(self) -> None:
        self.ax_3d.set_title("3D cloud")
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        self.ax_3d.view_init(elev=30.0, azim=-37.5)
        self._set_equal_3d(self.ax_3d, self.sample_points_raw)

        self.ax_map.set_title("Map view")
        self.ax_map.set_xlabel("Xh")
        self.ax_map.set_ylabel("Yh")
        self.ax_map.set_aspect("equal", adjustable="box")

    def _set_equal_3d(self, axis, points: np.ndarray) -> None:
        if points.size == 0:
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_zlim(0.0, 1.0)
            return
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        centre = 0.5 * (mins + maxs)
        radius = 0.5 * float(np.max(maxs - mins))
        if radius <= 0.0:
            radius = 1.0
        axis.set_xlim(centre[0] - radius, centre[0] + radius)
        axis.set_ylim(centre[1] - radius, centre[1] + radius)
        axis.set_zlim(centre[2] - radius, centre[2] + radius)

    def _recompute_labels(self) -> None:
        self.labels = _labels_from_instance_state(self.prepared.points_view, self.instance_state)

    def _append_instance(self, vertices_xy: list[list[float]]) -> int:
        next_id = 1 + max((int(item["id"]) for item in self.instance_state["instances"]), default=0)
        self.instance_state["instances"].append({"id": next_id, "vertices_xy": vertices_xy})
        self._recompute_labels()
        self.is_dirty = True
        return int(np.count_nonzero(self.labels == next_id))

    def _set_polygon_artists(self) -> None:
        for patch in self.instance_patches:
            patch.remove()
        self.instance_patches = []
        vmax = max(len(self.instance_state["instances"]), 1)
        for offset, item in enumerate(self.instance_state["instances"], start=1):
            polygon = np.asarray(item["vertices_xy"], dtype=np.float64)
            rgba = cm.get_cmap("tab20")(Normalize(vmin=1.0, vmax=float(vmax))(float(offset)))
            patch = PolygonPatch(
                polygon,
                closed=True,
                facecolor=rgba,
                edgecolor=rgba,
                alpha=0.18,
                linewidth=1.5,
            )
            self.ax_map.add_patch(patch)
            self.instance_patches.append(patch)

    def _refresh_views(self, status: str) -> None:
        rgba = self._sample_rgba()
        self.scatter_3d.set_color(rgba)
        self.scatter_map.set_facecolors(rgba)
        self._set_polygon_artists()

        instance_count = len(self.instance_state["instances"])
        labelled_points = int(np.count_nonzero(self.labels > 0))
        unknown_points = int(np.count_nonzero(self.labels <= 0))
        next_id = 1 + max((int(item["id"]) for item in self.instance_state["instances"]), default=0)
        self.count_text.set_text(
            f"Instances: {instance_count}\n"
            f"Labelled points: {labelled_points}\n"
            f"Unknown points: {unknown_points}\n"
            f"Next instance id: {next_id}\n"
            f"Display points: {self.prepared.sample_indices.size}"
        )
        self.status_text.set_text(status)
        self.fig.canvas.draw_idle()

    def _on_polygon_complete(self, vertices: list[tuple[float, float]]) -> None:
        vertices_xy = [[float(x), float(y)] for x, y in vertices]
        selected_count = self._append_instance(vertices_xy)
        next_id = max((int(item["id"]) for item in self.instance_state["instances"]), default=0)
        self._restart_polygon_selector()
        self._refresh_views(f"Instance {next_id} added with {selected_count} points.")

    def _on_save_click(self, _event) -> None:
        self.save_result = save_manual_instance_session(self.prepared, self.labels, self.paths, self.instance_state)
        self.is_dirty = False
        self._refresh_views(f"Saved instances to {self.paths.session_dir}.")

    def _on_undo_click(self, _event) -> None:
        if not self.instance_state["instances"]:
            self._refresh_views("No instance to remove.")
            return
        removed = self.instance_state["instances"].pop()
        self._recompute_labels()
        self.is_dirty = True
        self._restart_polygon_selector()
        self._refresh_views(f"Removed instance {int(removed['id'])}.")

    def _on_reset_click(self, _event) -> None:
        self.instance_state = _default_instance_state()
        self._recompute_labels()
        self.is_dirty = True
        self._restart_polygon_selector()
        self._refresh_views("Cleared all instances.")

    def _on_key_press(self, event) -> None:
        if event.key == "u":
            self._on_undo_click(event)
        elif event.key == "r":
            self._on_reset_click(event)
        elif event.key == "s":
            self._on_save_click(event)
        elif event.key == "q":
            plt.close(self.fig)

    def _on_close(self, _event) -> None:
        if self.save_result is None or self.is_dirty:
            self.save_result = save_manual_instance_session(self.prepared, self.labels, self.paths, self.instance_state)
            self.is_dirty = False

    def run(self) -> ManualLabelResult:
        plt.show()
        if self.save_result is None:
            self.save_result = save_manual_instance_session(self.prepared, self.labels, self.paths, self.instance_state)
        return self.save_result


def launch_manual_label_gui(
    pointcloud_path: str | Path,
    *,
    output_root: str | Path = "pe3d_labels",
    max_display_points: int = 120_000,
    seed: int = 0,
    rotate_horizontal: bool = True,
    detrend: bool = False,
) -> ManualLabelResult:
    if matplotlib.get_backend().lower().endswith("agg"):
        raise RuntimeError(
            "Manual labeling requires an interactive matplotlib backend. "
            "Run pe3d label from a desktop session with Tk, Qt, or macOS backend support."
        )

    prepared, paths, labels, polygon_state = prepare_manual_label_data(
        pointcloud_path,
        output_root=output_root,
        max_display_points=max_display_points,
        seed=seed,
        rotate_horizontal=rotate_horizontal,
        detrend=detrend,
    )
    app = ManualLabelApp(prepared, paths, labels, polygon_state)
    return app.run()


def launch_manual_instance_gui(
    pointcloud_path: str | Path,
    *,
    output_root: str | Path = "pe3d_labels",
    max_display_points: int = 120_000,
    seed: int = 0,
    rotate_horizontal: bool = True,
    detrend: bool = False,
) -> ManualLabelResult:
    if matplotlib.get_backend().lower().endswith("agg"):
        raise RuntimeError(
            "Manual labeling requires an interactive matplotlib backend. "
            "Run pe3d label-instances from a desktop session with Tk, Qt, or macOS backend support."
        )

    prepared, paths, labels, instance_state = prepare_manual_instance_data(
        pointcloud_path,
        output_root=output_root,
        max_display_points=max_display_points,
        seed=seed,
        rotate_horizontal=rotate_horizontal,
        detrend=detrend,
    )
    app = ManualInstanceApp(prepared, paths, labels, instance_state)
    return app.run()
