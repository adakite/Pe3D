from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pe3d", description="Python CLI for 3D pebble granulometry from point clouds")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full point-cloud workflow")
    run_parser.add_argument("--pointcloud", type=Path, required=True)
    run_parser.add_argument("--param-csv", type=Path, default=None)
    run_parser.add_argument("--output-root", type=Path, default=Path("pe3d_outputs"))
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--sizing-mode", choices=["ellipsoid", "block"], default="ellipsoid")
    run_parser.add_argument("--fit-strategy", choices=["pca_maxabs", "bounded_pca", "hybrid_direct"], default="hybrid_direct")
    run_parser.add_argument("--orientation-mode", choices=["atan2", "matlab"], default="atan2")
    run_parser.add_argument("--preprocess-mode", choices=["matlab", "fast"], default="matlab")
    run_parser.add_argument(
        "--ground-mode",
        choices=["none", "smrf"],
        default="none",
        help="Optional ground/background filter for block mode. 'smrf' applies a simple morphological ground model.",
    )
    run_parser.add_argument(
        "--smrf-cell-size",
        type=float,
        default=0.0,
        help="SMRF grid cell size in cloud units. Use 0 for automatic sizing from point spacing.",
    )
    run_parser.add_argument(
        "--smrf-max-window",
        type=float,
        default=12.0,
        help="Maximum SMRF opening window in cloud units.",
    )
    run_parser.add_argument(
        "--smrf-height-threshold",
        type=float,
        default=0.35,
        help="Base nonground height threshold above the SMRF surface.",
    )
    run_parser.add_argument(
        "--smrf-slope-threshold",
        type=float,
        default=0.75,
        help="Slope-dependent additive threshold multiplier for SMRF.",
    )
    run_parser.add_argument(
        "--max-ellipsoid-cloud-fraction",
        type=float,
        default=0.25,
        help="Reject ellipsoids whose major diameter exceeds this fraction of the processed cloud max span. Use 0 to disable.",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker threads for per-grain fitting. Use 1 for serial, or 0 for auto.",
    )
    run_parser.add_argument("--skip-plots", action="store_true")
    run_parser.add_argument(
        "--save-grains",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export one .ply file per segmented grain. Disabled by default.",
    )
    run_parser.add_argument(
        "--save-colored-clouds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export original point cloud recolored by ellipsoid volume, elongation, and azimuth.",
    )
    run_parser.add_argument(
        "--block-split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental: split oversized merged block components with a local map-view watershed. Disabled by default.",
    )

    label_parser = subparsers.add_parser("label", help="Launch an interactive manual labeling GUI")
    label_parser.add_argument("--pointcloud", type=Path, required=True)
    label_parser.add_argument("--output-root", type=Path, default=Path("pe3d_labels"))
    label_parser.add_argument("--seed", type=int, default=0)
    label_parser.add_argument("--max-display-points", type=int, default=120_000)
    label_parser.add_argument(
        "--reset-session",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete any existing saved labeling session for this dataset before opening the GUI.",
    )
    label_parser.add_argument(
        "--rotate-horizontal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit a plane and rotate the cloud so the dominant surface becomes horizontal.",
    )
    label_parser.add_argument(
        "--detrend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the existing quadratic detrend after the horizontal rotation.",
    )

    instance_parser = subparsers.add_parser("label-instances", help="Launch an interactive manual instance-labeling GUI")
    instance_parser.add_argument("--pointcloud", type=Path, required=True)
    instance_parser.add_argument("--output-root", type=Path, default=Path("pe3d_labels"))
    instance_parser.add_argument("--seed", type=int, default=0)
    instance_parser.add_argument("--max-display-points", type=int, default=120_000)
    instance_parser.add_argument(
        "--reset-session",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete any existing saved instance-label session for this dataset before opening the GUI.",
    )
    instance_parser.add_argument(
        "--rotate-horizontal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit a plane and rotate the cloud so the dominant surface becomes horizontal.",
    )
    instance_parser.add_argument(
        "--detrend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the existing quadratic detrend after the horizontal rotation.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        result = run_pipeline(
            args.pointcloud,
            param_csv=args.param_csv,
            output_root=args.output_root,
            save_plots=not args.skip_plots,
            save_grains=args.save_grains,
            save_colored_clouds=args.save_colored_clouds,
            seed=args.seed,
            sizing_mode=args.sizing_mode,
            fit_strategy=args.fit_strategy,
            orientation_mode=args.orientation_mode,
            preprocess_mode=args.preprocess_mode,
            max_ellipsoid_cloud_fraction=args.max_ellipsoid_cloud_fraction,
            workers=args.workers,
            block_split=args.block_split,
            ground_mode=args.ground_mode,
            smrf_cell_size=args.smrf_cell_size,
            smrf_max_window=args.smrf_max_window,
            smrf_height_threshold=args.smrf_height_threshold,
            smrf_slope_threshold=args.smrf_slope_threshold,
        )
        summary = {
            "pointcloud": str(result.params.pointcloud_path),
            "labels_count": result.labels_count,
            "ellipsoid_count": result.ellipsoid_count,
            "accepted_ellipsoid_count": result.accepted_ellipsoid_count,
            "sizing_mode": result.sizing_mode,
            "csv_path": str(result.csv_path) if result.csv_path else None,
            "xlsx_path": str(result.xlsx_path) if result.xlsx_path else None,
            "colored_cloud_paths": {key: str(path) for key, path in result.colorized_cloud_paths.items()},
            "grain_dir": str(result.run_paths.grain_dir),
            "figure_dir": str(result.run_paths.figure_dir),
            "cloud_dir": str(result.run_paths.cloud_dir),
            "save_grains": bool(args.save_grains),
            "save_colored_clouds": bool(args.save_colored_clouds),
            "fit_strategy": result.fit_strategy,
            "orientation_mode": result.orientation_mode,
            "preprocess_mode": result.preprocess_mode,
            "max_ellipsoid_cloud_fraction": result.max_ellipsoid_cloud_fraction,
            "workers": result.workers,
            "block_split": bool(args.block_split),
            "ground_mode": result.ground_mode,
            "smrf_cell_size": float(args.smrf_cell_size),
            "smrf_max_window": float(args.smrf_max_window),
            "smrf_height_threshold": float(args.smrf_height_threshold),
            "smrf_slope_threshold": float(args.smrf_slope_threshold),
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "label":
        from .manual_label import clear_manual_label_session, launch_manual_label_gui

        if args.reset_session:
            clear_manual_label_session(args.pointcloud, output_root=args.output_root)

        result = launch_manual_label_gui(
            args.pointcloud,
            output_root=args.output_root,
            seed=args.seed,
            max_display_points=args.max_display_points,
            rotate_horizontal=args.rotate_horizontal,
            detrend=args.detrend,
        )
        summary = {
            "pointcloud": str(result.pointcloud_path),
            "session_dir": str(result.session_dir),
            "labels_path": str(result.labels_path),
            "metadata_path": str(result.metadata_path),
            "polygons_path": str(result.polygons_path),
            "preview_ply_path": str(result.preview_ply_path),
            "block_count": result.block_count,
            "non_block_count": result.non_block_count,
            "unknown_count": result.unknown_count,
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "label-instances":
        from .manual_label import clear_manual_label_session, launch_manual_instance_gui

        if args.reset_session:
            clear_manual_label_session(args.pointcloud, output_root=args.output_root, session_prefix="manual_instances")

        result = launch_manual_instance_gui(
            args.pointcloud,
            output_root=args.output_root,
            seed=args.seed,
            max_display_points=args.max_display_points,
            rotate_horizontal=args.rotate_horizontal,
            detrend=args.detrend,
        )
        summary = {
            "pointcloud": str(result.pointcloud_path),
            "session_dir": str(result.session_dir),
            "labels_path": str(result.labels_path),
            "metadata_path": str(result.metadata_path),
            "polygons_path": str(result.polygons_path),
            "preview_ply_path": str(result.preview_ply_path),
            "instance_count": result.instance_count,
            "labelled_point_count": result.block_count,
            "unknown_count": result.unknown_count,
        }
        print(json.dumps(summary, indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
