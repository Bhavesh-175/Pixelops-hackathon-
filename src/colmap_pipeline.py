"""
colmap_pipeline.py
===================
STAGE 4 (sparse SfM) and STAGE 5 (dense MVS) - thin, careful subprocess
wrappers around the real COLMAP CLI. No geometry, poses, or depth values
are computed in Python here; COLMAP does all of the actual photogrammetry.
This file only orchestrates the calls, validates preconditions, and parses
COLMAP's own outputs for diagnostics.

Requires: COLMAP >= 3.8 on PATH (see README for install instructions).
Dense stereo (patch_match_stereo) requires a CUDA build of COLMAP + an
NVIDIA GPU. If unavailable, dense/mesh stages are skipped with a clear
message rather than faked.
"""

from __future__ import annotations
import json
import sqlite3
from pathlib import Path

import config
from utils import log, run_cmd, PipelineStageError, gpu_available


# ----------------------------------------------------------------------
# Stage 4a: feature extraction
# ----------------------------------------------------------------------
def extract_features(colmap_bin: str) -> None:
    cmd = [
        colmap_bin, "feature_extractor",
        "--database_path", str(config.WORKSPACE_DB),
        "--image_path", str(config.WORKSPACE_IMAGES),
        "--ImageReader.single_camera_per_folder", "0",
        "--ImageReader.camera_model", config.CAMERA_MODEL,
        "--SiftExtraction.use_gpu", "1" if gpu_available() else "0",
    ]
    run_cmd(cmd, "feature_extractor")


# ----------------------------------------------------------------------
# Stage 4b: matching
# ----------------------------------------------------------------------
def match_features(colmap_bin: str) -> None:
    use_gpu = "1" if gpu_available() else "0"
    if config.MATCHER_TYPE == "sequential":
        cmd = [
            colmap_bin, "sequential_matcher",
            "--database_path", str(config.WORKSPACE_DB),
            "--SequentialMatching.overlap", str(config.SEQUENTIAL_OVERLAP),
            "--SiftMatching.use_gpu", use_gpu,
        ]
    else:
        cmd = [
            colmap_bin, "exhaustive_matcher",
            "--database_path", str(config.WORKSPACE_DB),
            "--SiftMatching.use_gpu", use_gpu,
        ]
    run_cmd(cmd, "matcher")


# ----------------------------------------------------------------------
# Stage 4c: sparse reconstruction (mapper)
# ----------------------------------------------------------------------
def run_mapper(colmap_bin: str) -> Path:
    config.WORKSPACE_SPARSE.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_bin, "mapper",
        "--database_path", str(config.WORKSPACE_DB),
        "--image_path", str(config.WORKSPACE_IMAGES),
        "--output_path", str(config.WORKSPACE_SPARSE),
    ]
    run_cmd(cmd, "mapper", timeout=None)

    models = sorted(p for p in config.WORKSPACE_SPARSE.iterdir() if p.is_dir())
    if not models:
        raise PipelineStageError(
            "COLMAP mapper produced no reconstruction models at all. This means the "
            "feature matches did not chain into any registrable camera - check that "
            "the footage has genuine visual overlap between frames."
        )
    if len(models) > 1:
        log.warning(
            f"COLMAP produced {len(models)} DISCONNECTED reconstruction components "
            f"(model_0 .. model_{len(models)-1}). This typically means the footage has "
            f"a gap in overlap (e.g. the drone flew out of view and back, or two of the "
            f"four videos don't share any common viewpoint). Using the LARGEST component "
            f"({models[0].name} vs siblings) for downstream stages; inspect the smaller "
            f"components manually if they matter for your target."
        )
    # pick the model with the most images (largest reconstructed component)
    def _num_images(model_dir: Path) -> int:
        bin_path = model_dir / "images.bin"
        txt_path = model_dir / "images.txt"
        if bin_path.exists():
            return bin_path.stat().st_size  # proxy: bigger file ~ more images/points
        if txt_path.exists():
            return sum(1 for line in txt_path.read_text().splitlines() if line and not line.startswith("#"))
        return 0

    best_model = max(models, key=_num_images)
    log.info(f"Selected sparse model: {best_model}")
    return best_model


def sparse_stats(colmap_bin: str, model_dir: Path) -> dict:
    """Convert the binary model to TXT + PLY and parse real statistics."""
    txt_dir = config.SPARSE_DIR / "model_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([
        colmap_bin, "model_converter",
        "--input_path", str(model_dir),
        "--output_path", str(txt_dir),
        "--output_type", "TXT",
    ], "model_converter_txt")

    sparse_ply = config.SPARSE_DIR / "sparse_points.ply"
    run_cmd([
        colmap_bin, "model_converter",
        "--input_path", str(model_dir),
        "--output_path", str(sparse_ply),
        "--output_type", "PLY",
    ], "model_converter_ply")

    cameras_txt = txt_dir / "cameras.txt"
    images_txt = txt_dir / "images.txt"
    points_txt = txt_dir / "points3D.txt"

    n_cameras = sum(1 for l in cameras_txt.read_text().splitlines() if l and not l.startswith("#"))
    image_lines = [l for l in images_txt.read_text().splitlines() if l and not l.startswith("#")]
    n_registered_images = len(image_lines) // 2  # images.txt alternates pose line / points line
    n_points = sum(1 for l in points_txt.read_text().splitlines() if l and not l.startswith("#"))

    # mean track length / reprojection error, straight from points3D.txt columns:
    # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[...]
    errors = []
    for line in points_txt.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8:
            try:
                errors.append(float(parts[7]))
            except ValueError:
                pass
    mean_error = round(sum(errors) / len(errors), 4) if errors else None

    stats = {
        "sparse_model_dir": str(model_dir),
        "sparse_ply": str(sparse_ply),
        "num_cameras": n_cameras,
        "num_registered_images": n_registered_images,
        "num_sparse_points": n_points,
        "mean_reprojection_error_px": mean_error,
    }
    log.info(f"Sparse SfM stats: {stats}")

    if n_registered_images < 5:
        raise PipelineStageError(
            f"Only {n_registered_images} images registered by SfM - not enough for a "
            f"meaningful multi-view reconstruction. Check feature-match counts in "
            f"{config.WORKSPACE_DB} (open with any SQLite browser) to diagnose."
        )
    return stats


# ----------------------------------------------------------------------
# Stage 5: dense MVS
# ----------------------------------------------------------------------
def run_dense(colmap_bin: str, sparse_model: Path) -> dict:
    if not gpu_available():
        log.warning(
            "No NVIDIA GPU detected (nvidia-smi not found/failed). COLMAP's "
            "patch_match_stereo requires a CUDA build and GPU - dense MVS, depth "
            "maps, confidence maps and the mesh will be SKIPPED rather than faked. "
            "The sparse point cloud is still a genuine result and will be delivered."
        )
        return {"dense_available": False, "reason": "no_gpu"}

    config.WORKSPACE_DENSE.mkdir(parents=True, exist_ok=True)

    run_cmd([
        colmap_bin, "image_undistorter",
        "--image_path", str(config.WORKSPACE_IMAGES),
        "--input_path", str(sparse_model),
        "--output_path", str(config.WORKSPACE_DENSE),
        "--output_type", "COLMAP",
    ], "image_undistorter", timeout=None)

    run_cmd([
        colmap_bin, "patch_match_stereo",
        "--workspace_path", str(config.WORKSPACE_DENSE),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true" if config.PATCH_MATCH_GEOM_CONSISTENCY else "false",
        "--PatchMatchStereo.window_radius", str(config.PATCH_MATCH_WINDOW_RADIUS),
        "--PatchMatchStereo.num_samples", str(config.PATCH_MATCH_NUM_SAMPLES),
    ], "patch_match_stereo", timeout=None)

    fused_ply = config.WORKSPACE_DENSE / "fused.ply"
    run_cmd([
        colmap_bin, "stereo_fusion",
        "--workspace_path", str(config.WORKSPACE_DENSE),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused_ply),
    ], "stereo_fusion", timeout=None)

    if not fused_ply.exists() or fused_ply.stat().st_size == 0:
        raise PipelineStageError(
            "stereo_fusion produced an empty fused.ply. Dense reconstruction failed - "
            "likely insufficient geometric consistency between views. See "
            f"{config.WORKSPACE_DENSE}/stereo for per-image depth diagnostics."
        )

    depth_dir = config.WORKSPACE_DENSE / "stereo" / "depth_maps"
    n_depth_maps = len(list(depth_dir.glob("*.geometric.bin"))) if depth_dir.exists() else 0

    result = {
        "dense_available": True,
        "fused_ply": str(fused_ply),
        "workspace_dense": str(config.WORKSPACE_DENSE),
        "num_geometric_depth_maps": n_depth_maps,
    }
    log.info(f"Dense MVS stats: {result}")
    return result
