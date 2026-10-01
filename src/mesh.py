"""
mesh.py
=======
STAGE 9 - Surface reconstruction.

Uses COLMAP's own `poisson_mesher` directly on the fused dense point
cloud (this is COLMAP's mature, citation-grade Poisson surface
reconstruction - not a from-scratch reimplementation, per project
requirements). Open3D is then used only for CLEANUP: dropping tiny
disconnected components that Poisson reconstruction typically produces
at the edges of sparse coverage (floating debris), never for generating
new geometry.

If dense MVS was unavailable (no GPU), meshing is skipped with a clear
explanation - a mesh from a sparse SfM cloud alone would be unreliable
and we do not fabricate one.

Output: outputs/mesh/<basename>_mesh.ply
        outputs/visualization/mesh_view.png
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import open3d as o3d

import config
from utils import log, run_cmd, PipelineStageError


def run(colmap_bin: str, dense_info: dict) -> dict:
    if not dense_info.get("dense_available"):
        log.warning(
            "Skipping mesh generation: dense MVS was unavailable, and meshing a bare "
            "sparse SfM point cloud would not represent real surface geometry. "
            "Delivering point cloud + depth/confidence outputs only for this run."
        )
        return {"mesh_available": False, "reason": "no_dense_reconstruction"}

    fused_ply = Path(dense_info["fused_ply"])
    raw_mesh_ply = config.MESH_DIR / "poisson_raw.ply"

    run_cmd([
        colmap_bin, "poisson_mesher",
        "--input_path", str(fused_ply),
        "--output_path", str(raw_mesh_ply),
        "--PoissonMeshing.depth", str(config.POISSON_DEPTH),
        "--PoissonMeshing.trim", str(config.MESH_TRIM),
    ], "poisson_mesher", timeout=None)

    if not raw_mesh_ply.exists() or raw_mesh_ply.stat().st_size == 0:
        raise PipelineStageError(
            "colmap poisson_mesher produced an empty mesh file. This can happen if the "
            "fused point cloud lacks reliable normals/coverage - inspect "
            f"{fused_ply} directly to confirm it has enough points."
        )

    mesh = o3d.io.read_triangle_mesh(str(raw_mesh_ply))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise PipelineStageError(
            f"Mesh loaded from {raw_mesh_ply} has 0 vertices or 0 triangles - Poisson "
            f"reconstruction failed despite producing a file."
        )
    n_raw_verts, n_raw_tris = len(mesh.vertices), len(mesh.triangles)
    log.info(f"Raw Poisson mesh: {n_raw_verts} vertices, {n_raw_tris} triangles")

    # --- cleanup only: drop tiny disconnected components (floating debris) ---
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) > 0:
        largest = cluster_n_triangles.max()
        keep_mask = cluster_n_triangles[triangle_clusters] >= largest * config.MIN_MESH_COMPONENT_FRACTION
        mesh.remove_triangles_by_mask(~keep_mask)
        mesh.remove_unreferenced_vertices()
        n_components_dropped = int(np.sum(cluster_n_triangles < largest * config.MIN_MESH_COMPONENT_FRACTION))
        if n_components_dropped:
            log.info(f"Dropped {n_components_dropped} small disconnected mesh component(s) as debris.")

    mesh.compute_vertex_normals()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise PipelineStageError("Mesh cleanup removed every component - MIN_MESH_COMPONENT_FRACTION too aggressive.")

    out_mesh = config.MESH_DIR / f"{config.OUTPUT_BASENAME}_mesh.ply"
    o3d.io.write_triangle_mesh(str(out_mesh), mesh)

    stats = {
        "mesh_available": True,
        "output_mesh": str(out_mesh),
        "raw_vertices": n_raw_verts,
        "raw_triangles": n_raw_tris,
        "final_vertices": len(mesh.vertices),
        "final_triangles": len(mesh.triangles),
        "method": "COLMAP poisson_mesher on stereo_fusion dense cloud; Open3D used only for degenerate/duplicate/small-component cleanup",
    }
    (config.MESH_DIR / "mesh_stats.json").write_text(json.dumps(stats, indent=2))

    _render_mesh_screenshot(mesh, config.VIS_DIR / "mesh_view.png")
    log.info(f"Mesh stage complete: {stats}")
    return stats


def _render_mesh_screenshot(mesh: o3d.geometry.TriangleMesh, out_path: Path) -> None:
    try:
        vis = o3d.visualization.Visualizer()
        created = vis.create_window(visible=False, width=1280, height=960)
        if not created:
            raise RuntimeError("Open3D could not create an off-screen window")
        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out_path))
        vis.destroy_window()
        log.info(f"Saved mesh screenshot: {out_path}")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Could not render mesh screenshot off-screen ({exc}); mesh PLY is unaffected.")
