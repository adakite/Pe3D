from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULTS = {
    "iplot": True,
    "saveplot": True,
    "denoise": True,
    "decimate": False,
    "minima": False,
    "rotdetrend": True,
    "clean": True,
    "gridbynumber": True,
    "savegranulo": True,
    "savegrain": True,
    "res": 0.002,
    "nscale": 4,
    "minscale": 0.04,
    "maxscale": 2.0,
    "nnptcloud": 20,
    "radfactor": 0.6,
    "maxangle1": 60.0,
    "maxangle2": 10.0,
    "minflatness": 0.1,
    "fitmethod": "direct",
    "aquality_thresh": 10.0,
    "mindiam": 0.04,
    "naxis": 2,
    "nmin": 50,
    "dx_gbn": 0.0,
}


@dataclass(slots=True)
class Parameters:
    pointcloud_path: Path
    pointcloud_name: str
    pointcloud_dir: Path
    iplot: bool
    saveplot: bool
    denoise: bool
    decimate: bool
    minima: bool
    rotdetrend: bool
    clean: bool
    gridbynumber: bool
    savegranulo: bool
    savegrain: bool
    res: float
    nscale: int
    minscale: float
    maxscale: float
    nnptcloud: int
    radfactor: float
    maxangle1: float
    maxangle2: float
    minflatness: float
    fitmethod: str
    aquality_thresh: float
    mindiam: float
    naxis: int
    minnpoint: int
    dx_gbn: float
    iscolor: bool = False


@dataclass(slots=True)
class RunPaths:
    output_root: Path
    figure_dir: Path
    grain_dir: Path
    excel_dir: Path


def _as_bool(value: str | bool | int | float | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_int(value: str | int | float | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def _as_float(value: str | int | float | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def load_parameters(pointcloud_path: str | Path, param_csv: str | Path | None = None) -> Parameters:
    pointcloud = Path(pointcloud_path)
    if param_csv is None:
        param_csv = pointcloud.parent / "param.csv"
    param_csv = Path(param_csv)

    row: dict[str, str] | None = None
    if param_csv.exists():
        with param_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for candidate in csv.DictReader(handle):
                if candidate.get("Name") == pointcloud.name:
                    row = candidate
                    break

    values = DEFAULTS.copy()
    if row:
        values.update(
            {
                "iplot": _as_bool(row.get("iplot"), values["iplot"]),
                "saveplot": _as_bool(row.get("saveplot"), values["saveplot"]),
                "denoise": _as_bool(row.get("denoise"), values["denoise"]),
                "decimate": _as_bool(row.get("decimate"), values["decimate"]),
                "minima": _as_bool(row.get("minima"), values["minima"]),
                "rotdetrend": _as_bool(row.get("rotdetrend"), values["rotdetrend"]),
                "clean": _as_bool(row.get("clean"), values["clean"]),
                "gridbynumber": _as_bool(row.get("gridbynumber"), values["gridbynumber"]),
                "savegranulo": _as_bool(row.get("savegranulo"), values["savegranulo"]),
                "savegrain": _as_bool(row.get("savegrain"), values["savegrain"]),
                "res": _as_float(row.get("res"), values["res"]),
                "nscale": _as_int(row.get("nscale"), values["nscale"]),
                "minscale": _as_float(row.get("minscale"), values["minscale"]),
                "maxscale": _as_float(row.get("maxscale"), values["maxscale"]),
                "nnptcloud": _as_int(row.get("nnptcloud"), values["nnptcloud"]),
                "radfactor": _as_float(row.get("cf"), values["radfactor"]),
                "maxangle1": _as_float(row.get("maxangle1"), values["maxangle1"]),
                "maxangle2": _as_float(row.get("maxangle2"), values["maxangle2"]),
                "minflatness": _as_float(row.get("minflatness"), values["minflatness"]),
                "fitmethod": row.get("fitmethod", values["fitmethod"]) or values["fitmethod"],
                "aquality_thresh": _as_float(row.get("Aquality_thresh"), values["aquality_thresh"]),
                "mindiam": _as_float(row.get("mindiam"), values["mindiam"]),
                "naxis": _as_int(row.get("naxis"), values["naxis"]),
                "nmin": _as_int(row.get("nmin"), values["nmin"]),
                "dx_gbn": _as_float(row.get("dx_gbn"), values["dx_gbn"]),
            }
        )

    return Parameters(
        pointcloud_path=pointcloud,
        pointcloud_name=pointcloud.name,
        pointcloud_dir=pointcloud.parent,
        iplot=bool(values["iplot"]),
        saveplot=bool(values["saveplot"]),
        denoise=bool(values["denoise"]),
        decimate=bool(values["decimate"]),
        minima=bool(values["minima"]),
        rotdetrend=bool(values["rotdetrend"]),
        clean=bool(values["clean"]),
        gridbynumber=bool(values["gridbynumber"]),
        savegranulo=bool(values["savegranulo"]),
        savegrain=bool(values["savegrain"]),
        res=float(values["res"]),
        nscale=int(values["nscale"]),
        minscale=float(values["minscale"]),
        maxscale=float(values["maxscale"]),
        nnptcloud=int(values["nnptcloud"]),
        radfactor=float(values["radfactor"]),
        maxangle1=float(values["maxangle1"]),
        maxangle2=float(values["maxangle2"]),
        minflatness=float(values["minflatness"]),
        fitmethod=str(values["fitmethod"]),
        aquality_thresh=float(values["aquality_thresh"]),
        mindiam=float(values["mindiam"]),
        naxis=int(values["naxis"]),
        minnpoint=max(int(values["nnptcloud"]), int(values["nmin"])),
        dx_gbn=float(values["dx_gbn"]),
    )


def create_run_paths(output_root: str | Path, pointcloud_name: str) -> RunPaths:
    output_root = Path(output_root)
    stem = Path(pointcloud_name).stem

    for run_index in range(1, 10_000):
        grain_dir = output_root / "Grain" / f"{stem}_n{run_index}"
        figure_dir = output_root / "Figure" / f"{stem}_n{run_index}"
        excel_dir = output_root / "Excel" / f"{stem}_n{run_index}"
        if not grain_dir.exists() and not figure_dir.exists() and not excel_dir.exists():
            grain_dir.mkdir(parents=True, exist_ok=True)
            figure_dir.mkdir(parents=True, exist_ok=True)
            excel_dir.mkdir(parents=True, exist_ok=True)
            return RunPaths(
                output_root=output_root,
                figure_dir=figure_dir,
                grain_dir=grain_dir,
                excel_dir=excel_dir,
            )

    raise RuntimeError("Could not allocate a new output directory.")

