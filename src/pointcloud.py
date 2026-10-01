"""
pointcloud.py
=============
STAGE 8 - Point cloud cleanup, statistics, visualization.

Loads the REAL dense point cloud COLMAP's stereo_fusion produced
(dense/fused.ply) - or, if the GPU/dense stage was unavailable, falls
back to the real sparse SfM point cloud so the project still produces a
genuine (if less dense) 3D result. Never fabricates points.

Output: outputs/pointcloud/<basename>.ply
        outputs/visualization/pointcloud_view.png
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import open3d as o3d

import config
from utils import log, PipelineStageError


def _load_source_cloud(dense_info: dict, sparse_stats: dict) -> tuple[o3d.geometry.PointCloud, str]:
    if dense_info.get("dense_available"):
        path = Path(dense_info["fused_ply"])
        source = "dense (COLMAP stereo_fusion)"
    else:
        path = Path(sparse_stats["sparse_ply"])
        source = "sparse (COLMAP SfM) - dense MVS was unavailable, see log"
        log.warning("Using SPARSE point cloud as the final cloud because dense MVS was skipped.")

    if not path.exists() or path.stat().st_size == 0:
        raise PipelineStageError(f"Point cloud source file missing or empty: {path}")

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise PipelineStageError(f"Point cloud loaded from {path} has zero points.")
    return pcd, source


def run(dense_info: dict, sparse: dict) -> dict:
    pcd, source = _load_source_cloud(dense_info, sparse)
    n_raw = len(pcd.points)
    log.info(f"Loaded {n_raw} points from {source}")

    if config.VOXEL_DOWNSAMPLE_SIZE > 0:
        pcd = pcd.voxel_down_sample(config.VOXEL_DOWNSAMPLE_SIZE)
        log.info(f"Voxel-downsampled to {len(pcd.points)} points (voxel={config.VOXEL_DOWNSAMPLE_SIZE})")

    pcd_clean, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=config.OUTLIER_NB_NEIGHBORS,
        std_ratio=config.OUTLIER_STD_RATIO,
    )
    n_clean = len(pcd_clean.points)
    n_removed = len(pcd.points) - n_clean
    log.info(f"Statistical outlier removal: kept {n_clean}, removed {n_removed}")

    if n_clean == 0:
        raise PipelineStageError("Outlier removal eliminated the entire point cloud - thresholds too strict.")

    if not pcd_clean.has_normals():
        pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    out_ply = config.POINTCLOUD_DIR / f"{config.OUTPUT_BASENAME}_pointcloud.ply"
    o3d.io.write_point_cloud(str(out_ply), pcd_clean)

    pts = np.asarray(pcd_clean.points)
    bbox_min = pts.min(axis=0).tolist()
    bbox_max = pts.max(axis=0).tolist()
    has_color = pcd_clean.has_colors()

    stats = {
        "source": source,
        "output_ply": str(out_ply),
        "raw_points": n_raw,
        "cleaned_points": n_clean,
        "removed_outliers": n_removed,
        "has_color": has_color,
        "bounding_box_min": bbox_min,
        "bounding_box_max": bbox_max,
        "extent": (np.array(bbox_max) - np.array(bbox_min)).tolist(),
        "note": (
            "Coordinates are in COLMAP's arbitrary reconstruction scale, not metres, "
            "unless ground control points / known measurements were supplied. See "
            "README 'Photogrammetry limitations'."
        ),
    }
    (config.POINTCLOUD_DIR / "pointcloud_stats.json").write_text(json.dumps(stats, indent=2))

    _render_screenshot(pcd_clean, config.VIS_DIR / "pointcloud_view.png")
    log.info(f"Point cloud stage complete: {stats}")
    return stats


def _render_screenshot(geometry: o3d.geometry.Geometry3D, out_path: Path) -> None:
    """Off-screen render for a documentation image (headless-safe)."""
    try:
        vis = o3d.visualization.Visualizer()
        created = vis.create_window(visible=False, width=1280, height=960)
        if not created:
            raise RuntimeError("Open3D could not create an off-screen window")
        vis.add_geometry(geometry)
        vis.get_render_option().point_size = 2.0
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out_path))
        vis.destroy_window()
        log.info(f"Saved point cloud screenshot: {out_path}")
    except Exception as exc:  # noqa: BLE001
        log.warning(
            f"Could not render an off-screen screenshot ({exc}). This is common on "
            f"headless/remote setups without a GPU display context. The PLY output "
            f"itself is unaffected - open it in the HTML viewer or any PLY viewer."
        )
