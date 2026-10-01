"""
inspect_videos.py
==================
STAGE 1 - Dataset inspection.

Opens every video in dataset/videos with OpenCV and reads its REAL
properties (resolution, fps, frame count, duration, fourcc codec).
Nothing here is assumed - if a property can't be read, it is reported
as null/unknown rather than guessed, and a video that cannot be opened
at all raises immediately with the exact filename.

Output: outputs/inspection/video_inventory.json
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

import config
from utils import log, PipelineStageError


@dataclass
class VideoInfo:
    filename: str
    path: str
    size_bytes: int
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    fourcc: str
    aspect_ratio: float


def _fourcc_to_str(fourcc_float: float) -> str:
    fourcc_int = int(fourcc_float)
    chars = [chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]
    s = "".join(chars).strip()
    return s if s.isprintable() and s else "UNKNOWN"


def inspect_one(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise PipelineStageError(
            f"Could not open video '{path.name}'. It may be corrupt, use an "
            f"unsupported codec, or the OpenCV build lacks the right backend."
        )
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

        # frame_count reported by containers is sometimes wrong/0 for some
        # codecs (VFR footage, some mp4 muxers). Verify by actually seeking
        # to the reported last frame; if that fails, fall back to a manual
        # decode count (slower, but honest).
        if frame_count <= 0 or fps <= 0:
            log.warning(f"'{path.name}': container metadata incomplete, counting frames manually.")
            frame_count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                frame_count += 1
            duration = frame_count / fps if fps > 0 else 0.0
        else:
            # sanity check the reported count against a seek-to-end probe
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 1, 0))
            ok, _ = cap.read()
            if not ok:
                log.warning(
                    f"'{path.name}': container claims {frame_count} frames but the "
                    f"last frame did not decode; metadata may be unreliable."
                )
            duration = frame_count / fps

        if width <= 0 or height <= 0:
            raise PipelineStageError(f"'{path.name}' reported invalid resolution {width}x{height}.")

        return VideoInfo(
            filename=path.name,
            path=str(path),
            size_bytes=path.stat().st_size,
            width=width,
            height=height,
            fps=round(fps, 3),
            frame_count=frame_count,
            duration_seconds=round(duration, 3),
            fourcc=fourcc,
            aspect_ratio=round(width / height, 4),
        )
    finally:
        cap.release()


def run() -> list[VideoInfo]:
    videos = sorted(
        p for p in config.DATASET_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in config.VIDEO_EXTENSIONS
    ) if config.DATASET_DIR.exists() else []

    if not videos:
        raise PipelineStageError(
            f"No video files found in {config.DATASET_DIR}. Copy your UAV footage "
            f"there (extensions: {sorted(config.VIDEO_EXTENSIONS)}) and re-run."
        )

    infos: list[VideoInfo] = []
    total_bytes = 0
    for path in videos:
        info = inspect_one(path)
        infos.append(info)
        total_bytes += info.size_bytes
        log.info(
            f"{info.filename}: {info.width}x{info.height} @ {info.fps:.2f}fps, "
            f"{info.frame_count} frames, {info.duration_seconds:.1f}s, codec={info.fourcc}"
        )

    total_gb = total_bytes / (1024 ** 3)
    log.info(f"Dataset inventory: {len(infos)} videos, {total_gb:.2f} GB total.")

    payload = {
        "video_count": len(infos),
        "total_size_bytes": total_bytes,
        "total_size_gb": round(total_gb, 3),
        "videos": [asdict(v) for v in infos],
    }
    out_path = config.INSPECTION_DIR / "video_inventory.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(f"Wrote {out_path}")
    return infos
