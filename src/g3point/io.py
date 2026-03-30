from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement

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


def load_pointcloud(path: str | Path) -> PointCloud:
    path = Path(path)
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    field_names = set(vertex.dtype.names or ())

    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64, copy=False)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    points -= np.nanmin(points, axis=0, keepdims=True)

    extra_fields: dict[str, np.ndarray] = {}
    for name in sorted(field_names - {"x", "y", "z"}):
        extra_fields[name] = np.asarray(vertex[name])[valid]

    return PointCloud(
        points=points,
        extra_fields=extra_fields,
        iscolor={"red", "green", "blue"}.issubset(field_names),
    )


def write_pointcloud(path: str | Path, points: np.ndarray) -> None:
    path = Path(path)
    vertices = np.empty(points.shape[0], dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    PlyData([PlyElement.describe(vertices, "vertex")], text=True).write(str(path))


def export_grains(points: np.ndarray, labels: np.ndarray, run_paths: RunPaths, pointcloud_name: str) -> list[Path]:
    output_paths: list[Path] = []
    positive_labels = np.unique(labels[labels > 0])
    for label in positive_labels:
        grain_points = points[labels == label]
        output_path = run_paths.grain_dir / f"{pointcloud_name}_grain_{int(label)}.ply"
        write_pointcloud(output_path, grain_points)
        output_paths.append(output_path)
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

