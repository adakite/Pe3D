from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from .config import RunPaths


GRANULO_COLUMNS = [
    "Ngrain",
    "Xc",
    "Yc",
    "Zc",
    "Dmax",
    "Dmed",
    "Dmin",
    "angle_Mview",
    "angle_Xview",
]


@dataclass(slots=True)
class PointCloud:
    points: np.ndarray
    extra_fields: dict[str, np.ndarray] = field(default_factory=dict)
    iscolor: bool = False

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


def _normalize_loaded_points(
    points: np.ndarray,
    extra_fields: dict[str, np.ndarray],
    *,
    iscolor: bool,
) -> PointCloud:
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    points -= np.nanmin(points, axis=0, keepdims=True)
    extra_fields = {name: np.asarray(values)[valid] for name, values in extra_fields.items()}

    return PointCloud(
        points=points,
        extra_fields=extra_fields,
        iscolor=iscolor,
    )


def _load_ply_pointcloud(path: Path) -> PointCloud:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    field_names = set(vertex.dtype.names or ())
    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64, copy=False)

    extra_fields: dict[str, np.ndarray] = {}
    for name in sorted(field_names - {"x", "y", "z"}):
        extra_fields[name] = np.asarray(vertex[name])

    return _normalize_loaded_points(
        points,
        extra_fields,
        iscolor={"red", "green", "blue"}.issubset(field_names),
    )


def _load_las_pointcloud(path: Path) -> PointCloud:
    try:
        import laspy
    except ImportError as exc:
        raise RuntimeError(
            "LAS/LAZ support requires 'laspy' with a LAZ backend. "
            "Install pe3d with its declared dependencies or add 'laspy[lazrs]'."
        ) from exc

    las = laspy.read(str(path))
    points = np.asarray(las.xyz, dtype=np.float64)
    dimension_names = list(las.point_format.dimension_names)
    dimension_lookup = {name.lower(): name for name in dimension_names}

    extra_fields: dict[str, np.ndarray] = {}
    for lower_name, source_name in sorted(dimension_lookup.items()):
        if lower_name in {"x", "y", "z"}:
            continue
        extra_fields[lower_name] = np.asarray(las[source_name])

    return _normalize_loaded_points(
        points,
        extra_fields,
        iscolor={"red", "green", "blue"}.issubset(dimension_lookup),
    )


def load_pointcloud(path: str | Path) -> PointCloud:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return _load_ply_pointcloud(path)
    if suffix in {".las", ".laz"}:
        return _load_las_pointcloud(path)
    raise ValueError(f"Unsupported point-cloud format: '{path.suffix}'. Expected .ply, .las, or .laz.")


def write_pointcloud(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None, *, text: bool = True) -> None:
    path = Path(path)
    dtype = [("x", "f8"), ("y", "f8"), ("z", "f8")]
    if colors is not None:
        dtype.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices = np.empty(points.shape[0], dtype=dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    if colors is not None:
        clipped = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 255.0).astype(np.uint8)
        vertices["red"] = clipped[:, 0]
        vertices["green"] = clipped[:, 1]
        vertices["blue"] = clipped[:, 2]
    PlyData([PlyElement.describe(vertices, "vertex")], text=text).write(str(path))


def export_grains(points: np.ndarray, labels: np.ndarray, run_paths: RunPaths, pointcloud_name: str) -> list[Path]:
    output_paths: list[Path] = []
    positive_labels = np.unique(labels[labels > 0])
    for label in positive_labels:
        grain_points = points[labels == label]
        output_path = run_paths.grain_dir / f"{pointcloud_name}_grain_{int(label)}.ply"
        write_pointcloud(output_path, grain_points)
        output_paths.append(output_path)
    return output_paths


def _colormap_rgb(
    values: np.ndarray,
    *,
    cmap_name: str,
    invalid_mask: np.ndarray | None = None,
    invalid_rgb: tuple[int, int, int] = (96, 96, 96),
) -> np.ndarray:
    cache_root = Path(tempfile.gettempdir()) / "pe3d-mpl-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    from matplotlib import cm
    from matplotlib.colors import Normalize

    values = np.asarray(values, dtype=np.float64)
    if invalid_mask is None:
        invalid_mask = ~np.isfinite(values)
    else:
        invalid_mask = np.asarray(invalid_mask, dtype=bool) | ~np.isfinite(values)

    rgb = np.empty((values.shape[0], 3), dtype=np.uint8)
    rgb[:] = np.asarray(invalid_rgb, dtype=np.uint8)
    valid_values = values[~invalid_mask]
    if valid_values.size == 0:
        return rgb

    vmin = float(np.min(valid_values))
    vmax = float(np.max(valid_values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(valid_values))
    rgb[~invalid_mask] = np.rint(255.0 * rgba[:, :3]).astype(np.uint8)
    return rgb


def _binary_rgb(
    mask: np.ndarray,
    *,
    true_rgb: tuple[int, int, int],
    false_rgb: tuple[int, int, int],
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    rgb = np.empty((mask.shape[0], 3), dtype=np.uint8)
    rgb[:] = np.asarray(false_rgb, dtype=np.uint8)
    rgb[mask] = np.asarray(true_rgb, dtype=np.uint8)
    return rgb


def export_colorized_clouds(
    raw_points: np.ndarray,
    processed_points: np.ndarray,
    labels: np.ndarray,
    ellipsoids: list[dict[str, object]],
    granulo: dict[str, np.ndarray],
    run_paths: RunPaths,
    pointcloud_name: str,
) -> dict[str, Path]:
    if raw_points.size == 0 or processed_points.size == 0 or labels.size == 0:
        return {}

    tree = cKDTree(processed_points)
    _, nearest_idx = tree.query(raw_points, k=1, workers=-1)
    raw_labels = labels[np.asarray(nearest_idx, dtype=np.int64)]

    label_count = max(int(labels.max()) if labels.size else 0, len(ellipsoids))
    volume_by_label = np.full(label_count + 1, np.nan, dtype=np.float64)
    elongation_by_label = np.full(label_count + 1, np.nan, dtype=np.float64)
    azimuth_by_label = np.full(label_count + 1, np.nan, dtype=np.float64)

    valid_indices = np.asarray(granulo.get("valid_indices", np.empty(0, dtype=np.int64)), dtype=np.int64)
    angle_map_deg = np.degrees(np.asarray(granulo.get("angle_Mview", np.empty(0)), dtype=np.float64))
    diameters = np.asarray(granulo.get("diameter", np.empty((3, 0))), dtype=np.float64)
    for position, ellipsoid_idx in enumerate(valid_indices.tolist()):
        label = int(ellipsoid_idx) + 1
        ellipsoid = ellipsoids[int(ellipsoid_idx)]
        volume_by_label[label] = math.log10(max(float(ellipsoid["V"]), 1e-12))
        if diameters.shape[1] > position:
            elongation_by_label[label] = 1.0 - float(diameters[2, position] / max(diameters[0, position], 1e-12))
        if angle_map_deg.size > position:
            azimuth_by_label[label] = float(angle_map_deg[position])

    safe_labels = np.clip(raw_labels.astype(np.int64, copy=False), 0, label_count)
    volume_values = volume_by_label[safe_labels]
    elongation_values = elongation_by_label[safe_labels]
    azimuth_values = azimuth_by_label[safe_labels]
    invalid = safe_labels <= 0

    stem = Path(pointcloud_name).stem
    output_paths = {
        "volume": run_paths.cloud_dir / f"{stem}_by_volume.ply",
        "elongation": run_paths.cloud_dir / f"{stem}_by_elongation.ply",
        "azimuth": run_paths.cloud_dir / f"{stem}_by_azimuth.ply",
    }
    write_pointcloud(output_paths["volume"], raw_points, _colormap_rgb(volume_values, cmap_name="viridis", invalid_mask=invalid), text=False)
    write_pointcloud(output_paths["elongation"], raw_points, _colormap_rgb(elongation_values, cmap_name="plasma", invalid_mask=invalid), text=False)
    write_pointcloud(output_paths["azimuth"], raw_points, _colormap_rgb(azimuth_values, cmap_name="twilight", invalid_mask=invalid), text=False)
    return output_paths


def export_smrf_clouds(
    raw_points: np.ndarray,
    processed_points: np.ndarray,
    height_above_ground: np.ndarray,
    nonground_mask: np.ndarray,
    run_paths: RunPaths,
    pointcloud_name: str,
) -> dict[str, Path]:
    if raw_points.size == 0 or processed_points.size == 0 or height_above_ground.size == 0:
        return {}

    tree = cKDTree(processed_points)
    _, nearest_idx = tree.query(raw_points, k=1, workers=-1)
    nearest_idx = np.asarray(nearest_idx, dtype=np.int64)
    raw_height = np.asarray(height_above_ground, dtype=np.float64)[nearest_idx]
    raw_nonground = np.asarray(nonground_mask, dtype=bool)[nearest_idx]

    stem = Path(pointcloud_name).stem
    output_paths = {
        "height_above_ground": run_paths.cloud_dir / f"{stem}_height_above_ground.ply",
        "ground_mask": run_paths.cloud_dir / f"{stem}_ground_mask.ply",
    }
    write_pointcloud(
        output_paths["height_above_ground"],
        raw_points,
        _colormap_rgb(raw_height, cmap_name="inferno", invalid_mask=~np.isfinite(raw_height)),
        text=False,
    )
    write_pointcloud(
        output_paths["ground_mask"],
        raw_points,
        _binary_rgb(raw_nonground, true_rgb=(244, 131, 66), false_rgb=(61, 92, 140)),
        text=False,
    )
    return output_paths


def export_granulo(run_paths: RunPaths, pointcloud_name: str, granulo: dict[str, np.ndarray]) -> tuple[Path, Path]:
    rows: list[list[float | int]] = []
    diameters = granulo.get("diameter", np.empty((3, 0)))
    locations = granulo.get("Location", np.empty((3, 0)))
    map_angles = np.degrees(granulo.get("angle_Mview", np.empty(0)))
    x_angles = np.degrees(granulo.get("angle_Xview", np.empty(0)))

    for idx in range(diameters.shape[1]):
        rows.append(
            [
                idx + 1,
                float(locations[0, idx]),
                float(locations[1, idx]),
                float(locations[2, idx]),
                float(diameters[0, idx]),
                float(diameters[1, idx]),
                float(diameters[2, idx]),
                float(map_angles[idx]),
                float(x_angles[idx]),
            ]
        )

    frame = pd.DataFrame(rows, columns=GRANULO_COLUMNS)
    csv_path = run_paths.excel_dir / f"{pointcloud_name}_granulo.csv"
    xlsx_path = run_paths.excel_dir / f"{pointcloud_name}_granulo.xlsx"
    frame.to_csv(csv_path, index=False, header=False, float_format="%.8g")
    frame.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path
