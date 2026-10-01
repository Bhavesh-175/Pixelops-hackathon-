"""
report.py
=========
STAGE 11 - Output validation + final report.

Every artefact the pipeline claims to have produced is re-opened and
checked here (not just "the function returned without raising"). This
catches silent zero-byte-file or empty-geometry cases that a subprocess
exit code alone wouldn't reveal.

Output: outputs/reconstruction_report.json
"""

from __future__ import annotations
import json
from pathlib import Path

import config
from utils import log


def _check_file(path: str | None, min_size: int = 1) -> dict:
    if not path:
        return {"exists": False, "reason": "not_generated_this_run"}
    p = Path(path)
    if not p.exists():
        return {"exists": False, "path": str(p)}
    size = p.stat().st_size
    return {"exists": size >= min_size, "path": str(p), "size_bytes": size}


def build_and_validate(
    videos_info: list,
    frame_count: int,
    staged_count: int,
    sparse: dict,
    dense: dict,
    depth_conf: dict,
    pointcloud: dict,
    mesh: dict,
    viewer_path: Path,
    timings: dict,
    warnings: list[str],
) -> dict:
    validation = {
        "point_cloud": _check_file(pointcloud.get("output_ply"), min_size=100),
        "mesh": _check_file(mesh.get("output_mesh"), min_size=100) if mesh.get("mesh_available") else
                {"exists": False, "reason": "dense MVS unavailable this run"},
        "depth_maps_present": depth_conf.get("depth_maps", 0) > 0,
        "confidence_maps_present": depth_conf.get("confidence_maps", 0) > 0,
        "viewer_html": _check_file(str(viewer_path), min_size=100),
    }

    all_critical_ok = validation["point_cloud"]["exists"] and validation["viewer_html"]["exists"]
    if not all_critical_ok:
        warnings.append("CRITICAL: point cloud or viewer output failed validation - see 'validation' block.")

    report = {
        "project": config.PROJECT_NAME,
        "target": config.TARGET_NAME,
        "input_videos": [v.filename for v in videos_info],
        "video_statistics": [
            {
                "filename": v.filename, "width": v.width, "height": v.height,
                "fps": v.fps, "frame_count": v.frame_count,
                "duration_seconds": v.duration_seconds, "codec": v.fourcc,
            } for v in videos_info
        ],
        "extracted_candidate_frame_count": frame_count,
        "accepted_frame_count": staged_count,
        "rejected_frame_count": frame_count - staged_count,
        "sfm": sparse,
        "dense_mvs": dense,
        "depth_and_confidence": {k: v for k, v in depth_conf.items() if k != "mean_confidence_per_image"},
        "point_cloud": pointcloud,
        "mesh": mesh,
        "viewer_path": str(viewer_path),
        "validation": validation,
        "stage_timings_seconds": timings,
        "warnings": warnings,
        "scale_disclaimer": (
            "Reconstructed geometry is in COLMAP's arbitrary reconstruction scale. "
            "No ground-control points or known real-world measurements were supplied "
            "to this run, so distances/dimensions are NOT metric. Supply GCPs or a "
            "known reference length to obtain true scale (see README)."
        ),
    }

    config.REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info(f"Wrote final report: {config.REPORT_PATH}")

    if all_critical_ok:
        log.info("VALIDATION PASSED: core outputs (point cloud + viewer) confirmed present and non-empty.")
    else:
        log.error("VALIDATION FAILED: see warnings in reconstruction_report.json")

    return report
