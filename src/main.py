"""
main.py
=======
ONE command to run the whole PixelOps pipeline:

    .\\.venv\\Scripts\\python.exe src\\main.py

Each stage is wrapped in utils.Timer + a try/except that stops the run
with a clear diagnostic instead of continuing on bad state. Re-running
this script is safe: workspace/images, workspace/sparse and
workspace/dense are rebuilt from scratch each run so results never mix
between different datasets or previous attempts.
"""

from __future__ import annotations
import argparse
import sys

import config
from utils import log, ensure_all_dirs, check_colmap_installed, PipelineStageError, Timer

import inspect_videos
import frame_selection
import quality
import colmap_pipeline
import depth_confidence
import pointcloud
import mesh as mesh_stage
import viewer
import report


def main() -> int:
    parser = argparse.ArgumentParser(description="PixelOps UAV photogrammetry pipeline")
    parser.add_argument("--skip-dense", action="store_true",
                        help="Force-skip dense MVS/mesh even if a GPU is available (sparse-only run).")
    args = parser.parse_args()

    timings: dict[str, float] = {}
    warnings: list[str] = []

    log.info(f"=== {config.PROJECT_NAME} :: {config.TARGET_NAME} ===")
    ensure_all_dirs()

    try:
        colmap_bin = check_colmap_installed()

        with Timer("1_inspect_videos", timings):
            videos_info = inspect_videos.run()

        with Timer("2_frame_selection", timings):
            frames = frame_selection.run(videos_info)

        with Timer("3_quality_filter", timings):
            staged_images = quality.run(frames)

        with Timer("4_feature_extraction", timings):
            colmap_pipeline.extract_features(colmap_bin)

        with Timer("4_feature_matching", timings):
            colmap_pipeline.match_features(colmap_bin)

        with Timer("4_sparse_mapping", timings):
            sparse_model_dir = colmap_pipeline.run_mapper(colmap_bin)

        with Timer("4_sparse_stats", timings):
            sparse_stats = colmap_pipeline.sparse_stats(colmap_bin, sparse_model_dir)

        with Timer("5_dense_mvs", timings):
            if args.skip_dense:
                log.warning("--skip-dense passed: forcing sparse-only run.")
                dense_info = {"dense_available": False, "reason": "user_requested_skip"}
                warnings.append("Dense MVS skipped by user flag --skip-dense.")
            else:
                dense_info = colmap_pipeline.run_dense(colmap_bin, sparse_model_dir)
                if not dense_info.get("dense_available"):
                    warnings.append(f"Dense MVS unavailable: {dense_info.get('reason')}")

        with Timer("6_7_depth_confidence", timings):
            depth_conf = depth_confidence.run(dense_info)

        with Timer("8_pointcloud", timings):
            pc_stats = pointcloud.run(dense_info, sparse_stats)

        with Timer("9_mesh", timings):
            mesh_stats = mesh_stage.run(colmap_bin, dense_info)
            if not mesh_stats.get("mesh_available"):
                warnings.append(f"Mesh unavailable: {mesh_stats.get('reason')}")

        with Timer("10_viewer", timings):
            viewer_path = viewer.run(mesh_stats, pc_stats)

        with Timer("11_report", timings):
            final_report = report.build_and_validate(
                videos_info=videos_info,
                frame_count=len(frames),
                staged_count=len(staged_images),
                sparse=sparse_stats,
                dense=dense_info,
                depth_conf=depth_conf,
                pointcloud=pc_stats,
                mesh=mesh_stats,
                viewer_path=viewer_path,
                timings=timings,
                warnings=warnings,
            )

    except PipelineStageError as exc:
        log.error(f"PIPELINE STOPPED: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception(f"UNEXPECTED ERROR: {exc}")
        return 2

    log.info("=== PIPELINE COMPLETE ===")
    log.info(f"Point cloud : {final_report['point_cloud'].get('output_ply')}")
    if final_report["mesh"].get("mesh_available"):
        log.info(f"Mesh        : {final_report['mesh'].get('output_mesh')}")
    log.info(f"360 viewer  : {final_report['viewer_path']}")
    log.info(f"Full report : {config.REPORT_PATH}")
    if warnings:
        log.warning(f"Completed with {len(warnings)} warning(s) - see reconstruction_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
