from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare import save_comparison
from .pipeline import run_pipeline
from .validation import validate_against_reference


def _default_pointcloud() -> Path:
    return Path("G3point/PointCloud/Otira_1cm_grains.ply")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="g3point", description="Python CLI for the G3Point workflow")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full point-cloud workflow")
    run_parser.add_argument("--pointcloud", type=Path, default=_default_pointcloud())
    run_parser.add_argument("--param-csv", type=Path, default=None)
    run_parser.add_argument("--output-root", type=Path, default=Path("python_outputs"))
    run_parser.add_argument("--reference-root", type=Path, default=Path("G3point"))
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--fit-strategy", choices=["pca_maxabs", "bounded_pca", "hybrid_direct"], default="hybrid_direct")
    run_parser.add_argument("--orientation-mode", choices=["atan2", "matlab"], default="atan2")
    run_parser.add_argument("--skip-validation", action="store_true")
    run_parser.add_argument("--skip-plots", action="store_true")
    run_parser.add_argument("--skip-grains", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Compare generated outputs to the bundled reference")
    validate_parser.add_argument("--generated-csv", type=Path, required=True)
    validate_parser.add_argument("--reference-csv", type=Path, required=True)
    validate_parser.add_argument("--generated-grain-dir", type=Path, required=True)
    validate_parser.add_argument("--reference-grain-dir", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare", help="Compare one or more candidate CSVs to a reference and save plots")
    compare_parser.add_argument("--reference-csv", type=Path, required=True)
    compare_parser.add_argument("--candidate", action="append", required=True, help="label=path/to/csv")
    compare_parser.add_argument("--output-dir", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {None, "run"}:
        result = run_pipeline(
            args.pointcloud,
            param_csv=args.param_csv,
            output_root=args.output_root,
            reference_root=None if args.skip_validation else args.reference_root,
            validate=not args.skip_validation,
            save_plots=not args.skip_plots,
            save_grains=not args.skip_grains,
            seed=args.seed,
            fit_strategy=args.fit_strategy,
            orientation_mode=args.orientation_mode,
        )
        summary = {
            "pointcloud": str(result.params.pointcloud_path),
            "labels_count": result.labels_count,
            "ellipsoid_count": result.ellipsoid_count,
            "accepted_ellipsoid_count": result.accepted_ellipsoid_count,
            "csv_path": str(result.csv_path) if result.csv_path else None,
            "xlsx_path": str(result.xlsx_path) if result.xlsx_path else None,
            "grain_dir": str(result.run_paths.grain_dir),
            "figure_dir": str(result.run_paths.figure_dir),
            "fit_strategy": result.fit_strategy,
            "orientation_mode": result.orientation_mode,
            "validation": result.validation.to_dict() if result.validation else None,
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "validate":
        summary = validate_against_reference(
            args.generated_csv,
            args.reference_csv,
            args.generated_grain_dir,
            args.reference_grain_dir,
        )
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    if args.command == "compare":
        candidates: dict[str, Path] = {}
        for item in args.candidate:
            if "=" not in item:
                parser.error(f"Invalid --candidate value: {item}")
            label, path = item.split("=", 1)
            candidates[label] = Path(path)
        summary_path = save_comparison(args.reference_csv, candidates, args.output_dir)
        print(json.dumps({"summary_path": str(summary_path)}, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
