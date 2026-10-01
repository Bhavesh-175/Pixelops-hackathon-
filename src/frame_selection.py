"""
frame_selection.py
===================
STAGE 2 - Intelligent frame extraction.

For each video, the extraction interval is DERIVED from that video's own
measured fps/duration/frame_count (from inspect_videos.py) - never a
hard-coded frame skip. Within the candidate frames, three cheap filters
run inline during decoding (so we never buffer the whole video):

  1. Minimum temporal spacing (config.MIN_FRAME_INTERVAL_SECONDS)
  2. Sharpness gate (variance of Laplacian) - drops obviously blurred frames
  3. Perceptual-hash near-duplicate rejection against the last ACCEPTED
     frame - drops frames the drone captured while nearly stationary

Frames are streamed to disk one at a time; the video is never fully
loaded into RAM.

Output: outputs/frames/<video_stem>/frame_XXXXXX.jpg
        outputs/frames/frame_manifest.json
"""

from __future__ import annotations
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

import config
from utils import log
from inspect_videos import VideoInfo


@dataclass
class FrameRecord:
    video: str
    frame_index: int
    timestamp_seconds: float
    output_path: str
    sharpness: float
    mean_brightness: float


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _phash(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Simple, dependency-free perceptual hash (DCT-free average hash variant)."""
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return diff.flatten()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _extraction_interval_frames(info: VideoInfo) -> int:
    """
    Derive a per-video frame-skip interval so that we land near
    config.TARGET_CANDIDATE_FRAMES_PER_VIDEO candidates, while never
    sampling closer than MIN_FRAME_INTERVAL_SECONDS apart.
    """
    if info.frame_count <= 0 or info.fps <= 0:
        return 1
    interval_by_target = max(1, info.frame_count // config.TARGET_CANDIDATE_FRAMES_PER_VIDEO)
    interval_by_time = max(1, round(config.MIN_FRAME_INTERVAL_SECONDS * info.fps))
    interval = max(interval_by_target, interval_by_time)
    return interval


def _process_video(info: VideoInfo, out_dir: Path) -> list[FrameRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    interval = _extraction_interval_frames(info)
    log.info(
        f"{info.filename}: extracting every {interval} frame(s) "
        f"(~{info.frame_count // interval if interval else 0} candidates)"
    )

    cap = cv2.VideoCapture(str(info.path))
    records: list[FrameRecord] = []
    last_accepted_hash: np.ndarray | None = None
    frame_idx = -1
    saved = 0
    rejected_blur = 0
    rejected_dupe = 0

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % interval != 0:
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = _sharpness(gray)
            brightness = float(gray.mean())

            if sharpness < config.MIN_SHARPNESS_VARIANCE:
                rejected_blur += 1
                continue
            if not (config.MIN_MEAN_BRIGHTNESS <= brightness <= config.MAX_MEAN_BRIGHTNESS):
                rejected_blur += 1
                continue

            phash = _phash(gray)
            if last_accepted_hash is not None:
                dist = _hamming(phash, last_accepted_hash)
                if dist < config.PHASH_MIN_HAMMING_DISTANCE:
                    rejected_dupe += 1
                    continue

            timestamp = frame_idx / info.fps if info.fps > 0 else 0.0
            out_name = f"frame_{frame_idx:06d}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            records.append(FrameRecord(
                video=info.filename,
                frame_index=frame_idx,
                timestamp_seconds=round(timestamp, 3),
                output_path=str(out_path),
                sharpness=round(sharpness, 2),
                mean_brightness=round(brightness, 2),
            ))
            last_accepted_hash = phash
            saved += 1
    finally:
        cap.release()

    log.info(
        f"{info.filename}: kept {saved} frames "
        f"(rejected {rejected_blur} blur/exposure, {rejected_dupe} near-duplicate)"
    )
    return records


def run(videos: list[VideoInfo]) -> list[FrameRecord]:
    if config.FRAMES_DIR.exists():
        shutil.rmtree(config.FRAMES_DIR)
    config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    all_records: list[FrameRecord] = []
    for info in videos:
        stem = Path(info.filename).stem
        video_out = config.FRAMES_DIR / stem
        all_records.extend(_process_video(info, video_out))

    manifest = {
        "strategy": (
            "Per-video adaptive temporal sampling: interval derived from each video's "
            "own frame_count/fps so every clip contributes a comparable number of "
            "candidates, gated by a minimum temporal spacing "
            f"({config.MIN_FRAME_INTERVAL_SECONDS}s) to bound viewpoint overlap. "
            "Sharpness (variance-of-Laplacian) and exposure filters reject blurred or "
            "over/under-exposed frames inline during decode. A running perceptual "
            "hash (average-hash, Hamming distance) rejects near-duplicate frames "
            "caused by the drone hovering, without ever holding more than one frame "
            "in memory at a time."
        ),
        "total_candidate_frames": len(all_records),
        "frames": [asdict(r) for r in all_records],
    }
    manifest_path = config.FRAMES_DIR / "frame_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(f"Total frames selected across all videos: {len(all_records)}")
    log.info(f"Wrote {manifest_path}")
    return all_records
