from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


COLUMNS = ["Ngrain", "Xc", "Yc", "Zc", "Dmax", "Dmed", "Dmin", "angle_Mview", "angle_Xview"]


def load_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=COLUMNS)


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.abs(a - b) % 180.0
    return np.minimum(delta, 180.0 - delta)


def matched_summary(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, object]:
    ref_xyz = reference[["Xc", "Yc", "Zc"]].to_numpy(dtype=float)
    cand_xyz = candidate[["Xc", "Yc", "Zc"]].to_numpy(dtype=float)
    cost = ((ref_xyz[:, None, :] - cand_xyz[None, :, :]) ** 2).sum(axis=2)
    ref_idx, cand_idx = linear_sum_assignment(cost)
    dist = np.sqrt(cost[ref_idx, cand_idx])

    matched_ref = reference.iloc[ref_idx].reset_index(drop=True)
    matched_cand = candidate.iloc[cand_idx].reset_index(drop=True)

    metrics: dict[str, dict[str, float]] = {}
    for col in ["Dmax", "Dmed", "Dmin"]:
        delta = matched_cand[col].to_numpy(float) - matched_ref[col].to_numpy(float)
        metrics[col] = {
            "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "bias": float(np.mean(delta)),
            "median_abs": float(np.median(np.abs(delta))),
            "p95_abs": float(np.percentile(np.abs(delta), 95)),
        }
    for col in ["angle_Mview", "angle_Xview"]:
        delta = angle_diff_deg(matched_cand[col].to_numpy(float), matched_ref[col].to_numpy(float))
        metrics[col] = {
            "mae": float(np.mean(delta)),
            "median_abs": float(np.median(delta)),
            "p95_abs": float(np.percentile(delta, 95)),
        }

    return {
        "rows": int(len(candidate)),
        "matched_rows": int(len(ref_idx)),
        "extra_rows": int(len(candidate) - len(ref_idx)),
        "center_distance": {
            "mean": float(np.mean(dist)),
            "median": float(np.median(dist)),
            "p95": float(np.percentile(dist, 95)),
            "max": float(np.max(dist)),
        },
        "metrics": metrics,
    }


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values)
    probs = np.linspace(0.0, 1.0, sorted_values.size, endpoint=True)
    return sorted_values, probs


def save_comparison(reference_csv: str | Path, candidates: dict[str, str | Path], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_table(reference_csv)
    candidate_tables = {label: load_table(path) for label, path in candidates.items()}

    summary = {
        "reference_csv": str(reference_csv),
        "reference_rows": int(len(reference)),
        "candidates": {label: matched_summary(reference, table) for label, table in candidate_tables.items()},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis_idx, col in enumerate(["Dmax", "Dmed", "Dmin"]):
        x, y = _cdf(reference[col].to_numpy(float))
        axes[axis_idx].plot(x, y, label="MATLAB", linewidth=2)
        for label, table in candidate_tables.items():
            x, y = _cdf(table[col].to_numpy(float))
            axes[axis_idx].plot(x, y, label=label, linewidth=1.6)
        axes[axis_idx].set_title(col)
        axes[axis_idx].set_xlabel("meters")
        axes[axis_idx].set_ylabel("CDF")
    axes[0].legend()
    fig.savefig(output_dir / "diameter_cdfs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis_idx, col in enumerate(["angle_Mview", "angle_Xview"]):
        x, y = _cdf(reference[col].to_numpy(float))
        axes[axis_idx].plot(x, y, label="MATLAB", linewidth=2)
        for label, table in candidate_tables.items():
            x, y = _cdf(table[col].to_numpy(float))
            axes[axis_idx].plot(x, y, label=label, linewidth=1.6)
        axes[axis_idx].set_title(col)
        axes[axis_idx].set_xlabel("degrees")
        axes[axis_idx].set_ylabel("CDF")
    axes[0].legend()
    fig.savefig(output_dir / "angle_cdfs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = list(candidate_tables)
    map_mae = [summary["candidates"][label]["metrics"]["angle_Mview"]["mae"] for label in labels]
    x_mae = [summary["candidates"][label]["metrics"]["angle_Xview"]["mae"] for label in labels]
    axes[0].bar(labels, map_mae)
    axes[0].set_title("Map Angle MAE")
    axes[0].set_ylabel("degrees")
    axes[1].bar(labels, x_mae)
    axes[1].set_title("X-view Angle MAE")
    fig.savefig(output_dir / "angle_mae.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    x = np.arange(3)
    for offset, label in enumerate(labels):
        maes = [summary["candidates"][label]["metrics"][col]["mae"] for col in ["Dmax", "Dmed", "Dmin"]]
        ax.bar(x + offset * width, maes, width=width, label=label)
    ax.set_xticks(x + width * (len(labels) - 1) / 2.0)
    ax.set_xticklabels(["Dmax", "Dmed", "Dmin"])
    ax.set_ylabel("MAE (m)")
    ax.legend()
    fig.savefig(output_dir / "diameter_mae.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return summary_path
