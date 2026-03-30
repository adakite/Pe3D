from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ValidationSummary:
    generated_rows: int
    reference_rows: int
    generated_grains: int
    reference_grains: int
    column_mae: list[float]
    column_max_abs: list[float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_against_reference(
    generated_csv: str | Path,
    reference_csv: str | Path,
    generated_grain_dir: str | Path,
    reference_grain_dir: str | Path,
) -> ValidationSummary:
    generated_frame = pd.read_csv(generated_csv, header=None)
    reference_frame = pd.read_csv(reference_csv, header=None)
    common_rows = min(len(generated_frame), len(reference_frame))
    common_cols = min(generated_frame.shape[1], reference_frame.shape[1])

    if common_rows:
        generated = generated_frame.iloc[:common_rows, :common_cols].to_numpy(dtype=float)
        reference = reference_frame.iloc[:common_rows, :common_cols].to_numpy(dtype=float)
        delta = generated - reference
        column_mae = np.mean(np.abs(delta), axis=0).tolist()
        column_max_abs = np.max(np.abs(delta), axis=0).tolist()
    else:
        column_mae = []
        column_max_abs = []

    generated_grains = len(list(Path(generated_grain_dir).glob("*.ply")))
    reference_grains = len(list(Path(reference_grain_dir).glob("*.ply")))
    return ValidationSummary(
        generated_rows=len(generated_frame),
        reference_rows=len(reference_frame),
        generated_grains=generated_grains,
        reference_grains=reference_grains,
        column_mae=column_mae,
        column_max_abs=column_max_abs,
    )
