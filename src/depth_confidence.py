"""
depth_confidence.py
====================
STAGE 6 - Depth map export
STAGE 7 - Confidence map derivation

COLMAP's patch_match_stereo (run with geom_consistency=true) writes, per
undistorted image, TWO independent depth estimates:

    <image>.photometric.bin   - depth from photometric (pixel-color) consistency alone
    <image>.geometric.bin     - depth from geometric consistency across ALL neighbour views

These are COLMAP's own native binary array format (a small text header
"width&height&channels&" followed by row-major float32 data), documented
in COLMAP's dense reconstruction source (mvs::Bitmap / DepthMap). We read
them directly - no reinterpretation, no synthetic values.

CONFIDENCE DEFINITION (explicit, so nothing here is a black box):
For every pixel with a valid geometric depth estimate, confidence is
1.0 minus the normalised relative disagreement between the geometric and
photometric depth estimates at that pixel:

    confidence = clamp(1 - |d_geom - d_photo| / d_geom, 0, 1)

Pixels where photometric and geometric stereo agree closely (the classic
signal that a surface point is genuinely well-observed from multiple
viewpoints) get HIGH confidence. Pixels where the two disagree strongly,
or where geometric stereo found no valid depth at all (occluded /
textureless / out of view), get LOW or zero confidence. This is a
standard multi-view-consistency confidence proxy, not a random or
cosmetic value.

Output: outputs/depth/<image>_depth.png        (16-bit, real metric-relative depth, colorized copy also saved)
        outputs/confidence/<image>_confidence.png (0-255, see formula above)
"""

from __future__ import annotations
import json
import struct
from pathlib import Path

import cv2
import numpy as np

import config
from utils import log, PipelineStageError


def read_colmap_array(path: Path) -> np.ndarray:
    """
    Parse COLMAP's dense mvs binary array format:
    ASCII header "WIDTH&HEIGHT&CHANNELS&" followed by little-endian float32
    data in (channels, height, width) storage order... COLMAP actually
    stores it column-major per-channel; we follow the documented reader
    logic used by colmap's own read_write_dense_model helpers.
    """
    with open(path, "rb") as f:
        header = b""
        while True:
            byte = f.read(1)
            if byte == b"&":
                header += byte
                if header.count(b"&") == 3:
                    break
            else:
                header += byte
        header_str = header.decode("ascii")
        width_s, height_s, channels_s, _ = header_str.split("&")
        width, height, channels = int(width_s), int(height_s), int(channels_s)

        data = np.fromfile(f, dtype=np.float32)
        expected = width * height * channels
        if data.size < expected:
            raise PipelineStageError(
                f"Depth/normal array {path.name} truncated: expected {expected} floats, got {data.size}."
            )
        data = data[:expected]
        # COLMAP stores arrays column-major (Fortran order) per channel
        array = data.reshape(channels, width, height).transpose(0, 2, 1)
        return array[0] if channels == 1 else array


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    d = depth.copy()
    lo, hi = np.percentile(d[valid], 2), np.percentile(d[valid], 98)
    d_clipped = np.clip(d, lo, hi)
    norm = np.zeros_like(d, dtype=np.uint8)
    norm[valid] = ((d_clipped[valid] - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)
    return colored


def run(dense_info: dict) -> dict:
    if not dense_info.get("dense_available"):
        log.warning("Dense stage unavailable - skipping depth/confidence export.")
        return {"depth_maps": 0, "confidence_maps": 0, "skipped_reason": dense_info.get("reason")}

    depth_map_dir = Path(dense_info["workspace_dense"]) / "stereo" / "depth_maps"
    geom_files = sorted(depth_map_dir.glob("*.geometric.bin"))
    if not geom_files:
        raise PipelineStageError(f"No geometric depth maps found in {depth_map_dir}.")

    n_depth, n_conf = 0, 0
    per_image_mean_conf = {}

    for geom_path in geom_files:
        base_name = geom_path.name.replace(".geometric.bin", "")
        photo_path = depth_map_dir / f"{base_name}.photometric.bin"

        depth_geom = read_colmap_array(geom_path)
        valid = depth_geom > 0

        # --- Stage 6: depth export (real geometric-consistency depth) ---
        depth_16 = np.zeros(depth_geom.shape, dtype=np.uint16)
        if np.any(valid):
            scale = 65535.0 / max(depth_geom[valid].max(), 1e-6)
            depth_16[valid] = (depth_geom[valid] * scale).astype(np.uint16)
        depth_png_path = config.DEPTH_DIR / f"{base_name}_depth.png"
        cv2.imwrite(str(depth_png_path), depth_16)
        vis_path = config.DEPTH_DIR / f"{base_name}_depth_colorized.png"
        cv2.imwrite(str(vis_path), _colorize_depth(depth_geom))
        n_depth += 1

        # --- Stage 7: confidence from geometric/photometric agreement ---
        confidence = np.zeros(depth_geom.shape, dtype=np.float32)
        if photo_path.exists():
            depth_photo = read_colmap_array(photo_path)
            both_valid = valid & (depth_photo > 0)
            rel_diff = np.zeros_like(depth_geom)
            rel_diff[both_valid] = np.abs(depth_geom[both_valid] - depth_photo[both_valid]) / depth_geom[both_valid]
            confidence[both_valid] = np.clip(1.0 - rel_diff[both_valid], 0.0, 1.0)
        else:
            log.warning(f"No photometric depth counterpart for {base_name}; confidence set to 0 there.")

        conf_8 = (confidence * 255).astype(np.uint8)
        conf_path = config.CONFIDENCE_DIR / f"{base_name}_confidence.png"
        cv2.imwrite(str(conf_path), conf_8)
        n_conf += 1
        if np.any(valid):
            per_image_mean_conf[base_name] = round(float(confidence[valid].mean()), 4)

    summary = {
        "depth_maps": n_depth,
        "confidence_maps": n_conf,
        "confidence_method": (
            "1 - |geometric_depth - photometric_depth| / geometric_depth, clamped [0,1], "
            "per pixel; computed from COLMAP patch_match_stereo's independent geometric- "
            "and photometric-consistency depth estimates. High confidence = the two "
            "independent stereo estimates agree; low/zero confidence = disagreement or "
            "no valid geometric depth (occlusion, textureless surface, out of view)."
        ),
        "mean_confidence_per_image": per_image_mean_conf,
    }
    (config.CONFIDENCE_DIR / "confidence_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Depth/confidence stage: {n_depth} depth maps, {n_conf} confidence maps.")
    return summary
