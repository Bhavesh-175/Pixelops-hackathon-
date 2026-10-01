"""
quality.py
==========
STAGE 3 - Final quality gate + COLMAP image staging.

frame_selection.py already applies inline sharpness/exposure gates while
decoding. This stage:

  1. Re-validates every selected frame can actually be read back
     (catches truncated/corrupt JPEG writes).
  2. Ranks and logs the full sharpness distribution so a human can sanity
     check the threshold used (transparency, not a black box).
  3. Copies survivors into workspace/images/ with GLOBALLY UNIQUE
     filenames ("<video_stem>__frame_XXXXXX.jpg") because COLMAP expects
     one flat directory - without the video-stem prefix, frame_000120.jpg
     from two different videos would collide and silently overwrite.

Output: outputs/quality/quality_report.json
        workspace/images/*.jpg  (COLMAP input)
"""

from __future__ import annotations
import json
import shutil
from pathlib import Path

import cv2

import config
from utils import log, PipelineStageError
from frame_selection import FrameRecord


def run(frames: list[FrameRecord]) -> list[Path]:
    if not frames:
        raise PipelineStageError("No frames survived Stage 2 (extraction). Cannot continue.")

    if config.WORKSPACE_IMAGES.exists():
        shutil.rmtree(config.WORKSPACE_IMAGES)
    config.WORKSPACE_IMAGES.mkdir(parents=True, exist_ok=True)

    staged: list[Path] = []
    corrupt: list[str] = []
    sharpness_values = []

    for rec in frames:
        src = Path(rec.output_path)
        img = cv2.imread(str(src))
        if img is None:
            corrupt.append(str(src))
            continue

        video_stem = Path(rec.video).stem
        dest_name = f"{video_stem}__{src.name}"
        dest = config.WORKSPACE_IMAGES / dest_name
        shutil.copy2(src, dest)
        staged.append(dest)
        sharpness_values.append(rec.sharpness)

    if corrupt:
        log.warning(f"{len(corrupt)} frames failed to re-decode and were skipped: {corrupt[:5]}...")

    if len(staged) < 8:
        raise PipelineStageError(
            f"Only {len(staged)} usable frames survived quality filtering - far too few "
            f"for multi-view SfM (need dozens at minimum with real overlap). Lower "
            f"MIN_SHARPNESS_VARIANCE in config.py or check the source footage for focus issues."
        )

    sharpness_values.sort()
    n = len(sharpness_values)
    stats = {
        "accepted": len(staged),
        "rejected_corrupt": len(corrupt),
        "sharpness_min": sharpness_values[0],
        "sharpness_median": sharpness_values[n // 2],
        "sharpness_max": sharpness_values[-1],
        "threshold_used": config.MIN_SHARPNESS_VARIANCE,
    }
    log.info(f"Quality stage: {stats}")

    report_path = config.QUALITY_DIR / "quality_report.json"
    report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info(f"Staged {len(staged)} images into {config.WORKSPACE_IMAGES}")
    return staged
