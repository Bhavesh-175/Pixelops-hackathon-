"""
config.py
=========
Single source of truth for every path, threshold and switch used by the
pipeline. This is the ONLY file you should need to edit to point PixelOps
at a new dataset / target.

Nothing in this file does any I/O by itself - it just defines constants
and derives paths. Directory creation happens in utils.ensure_all_dirs().
"""

from __future__ import annotations
from pathlib import Path
import shutil

# ----------------------------------------------------------------------
# 1. PROJECT IDENTITY  <-- EDIT THESE FOR YOUR DATASET
# ----------------------------------------------------------------------
PROJECT_NAME = "PixelOps"

# Free-text name of whatever you are reconstructing. Used only in logs,
# the HTML viewer title and the final report - it does not affect
# processing in any way.
TARGET_NAME = "Reconstruction Target"

# Output artefact base filename (no extension). Change this per dataset
# so re-runs on different datasets don't collide.
OUTPUT_BASENAME = "reconstruction"

# ----------------------------------------------------------------------
# 2. PATHS
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "dataset" / "videos"       # put your .mp4/.mov here
WORKSPACE_DIR = ROOT_DIR / "workspace"               # COLMAP database + images live here
OUTPUTS_DIR = ROOT_DIR / "outputs"

INSPECTION_DIR = OUTPUTS_DIR / "inspection"
FRAMES_DIR = OUTPUTS_DIR / "frames"
QUALITY_DIR = OUTPUTS_DIR / "quality"
SPARSE_DIR = OUTPUTS_DIR / "sparse"
DENSE_DIR = OUTPUTS_DIR / "dense"
DEPTH_DIR = OUTPUTS_DIR / "depth"
CONFIDENCE_DIR = OUTPUTS_DIR / "confidence"
POINTCLOUD_DIR = OUTPUTS_DIR / "pointcloud"
MESH_DIR = OUTPUTS_DIR / "mesh"
VIS_DIR = OUTPUTS_DIR / "visualization"
VIEWER_DIR = OUTPUTS_DIR / "viewer"
PANORAMA_DIR = OUTPUTS_DIR / "panorama"

# COLMAP working paths
WORKSPACE_IMAGES = WORKSPACE_DIR / "images"          # flat folder COLMAP consumes
WORKSPACE_DB = WORKSPACE_DIR / "database.db"
WORKSPACE_SPARSE = WORKSPACE_DIR / "sparse"          # colmap sparse model (binary)
WORKSPACE_DENSE = WORKSPACE_DIR / "dense"            # undistorted images + depth maps

REPORT_PATH = OUTPUTS_DIR / "reconstruction_report.json"
LOG_PATH = OUTPUTS_DIR / "pipeline.log"

ALL_DIRS = [
    DATASET_DIR, WORKSPACE_DIR, OUTPUTS_DIR,
    INSPECTION_DIR, FRAMES_DIR, QUALITY_DIR, SPARSE_DIR, DENSE_DIR,
    DEPTH_DIR, CONFIDENCE_DIR, POINTCLOUD_DIR, MESH_DIR, VIS_DIR,
    VIEWER_DIR, PANORAMA_DIR, WORKSPACE_IMAGES, WORKSPACE_DENSE,
]

# ----------------------------------------------------------------------
# 3. VIDEO / FRAME EXTRACTION
# ----------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# Target number of candidate frames to pull PER VIDEO before quality
# filtering. Chosen conservatively; actual survivors after quality/blur
# filtering will be fewer. If a video is short, fewer frames are taken.
TARGET_CANDIDATE_FRAMES_PER_VIDEO = 220

# Never sample closer together than this many seconds, regardless of fps,
# to avoid near-duplicate frames dominating the set.
MIN_FRAME_INTERVAL_SECONDS = 0.20

# Perceptual-duplicate rejection: frames whose difference hash differs by
# fewer than this many bits from the previously ACCEPTED frame are
# considered redundant and dropped.
PHASH_MIN_HAMMING_DISTANCE = 6

# ----------------------------------------------------------------------
# 4. IMAGE QUALITY FILTERING
# ----------------------------------------------------------------------
# Variance-of-Laplacian sharpness. Frames below this are blurred (motion
# blur / defocus) and rejected. This is a relative, not absolute, unit;
# main.py logs the full distribution so you can tune it per dataset.
MIN_SHARPNESS_VARIANCE = 40.0

# Mean pixel brightness (0-255) acceptable range - rejects near-black
# (exposure loss / lens covered) and near-white (blown-out sky) frames.
MIN_MEAN_BRIGHTNESS = 15.0
MAX_MEAN_BRIGHTNESS = 245.0

# ----------------------------------------------------------------------
# 5. COLMAP
# ----------------------------------------------------------------------
def _find_colmap() -> str:
    exe = shutil.which("colmap") or shutil.which("colmap.exe") or shutil.which("COLMAP.bat")
    return exe if exe else "colmap"  # fall back to PATH lookup at call time; utils validates it

COLMAP_BIN = _find_colmap()

# "sequential" is strongly preferred for UAV video because consecutive
# frames are already temporally/spatially ordered - it is far cheaper
# than exhaustive matching and gives better results for flight-path
# footage. Set to "exhaustive" for unordered photo sets.
MATCHER_TYPE = "sequential"          # "sequential" | "exhaustive"
SEQUENTIAL_OVERLAP = 10              # how many neighbouring frames to match against
CAMERA_MODEL = "SIMPLE_RADIAL"       # single unknown UAV camera, no calibration file
SINGLE_CAMERA_PER_VIDEO = True       # each video = one physical camera/lens

# Dense stereo requires an NVIDIA GPU + CUDA-enabled COLMAP build.
# The pipeline auto-detects this (see utils.gpu_available) and will
# clearly report + skip dense/mesh stages rather than fake results if
# unavailable.
PATCH_MATCH_WINDOW_RADIUS = 5
PATCH_MATCH_NUM_SAMPLES = 15
PATCH_MATCH_GEOM_CONSISTENCY = True

# ----------------------------------------------------------------------
# 6. POINT CLOUD CLEANUP
# ----------------------------------------------------------------------
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0
VOXEL_DOWNSAMPLE_SIZE = 0.0   # 0 = disabled; set e.g. 0.01 to downsample dense clouds

# ----------------------------------------------------------------------
# 7. MESH
# ----------------------------------------------------------------------
POISSON_DEPTH = 10                    # COLMAP poisson_mesher --PoissonMeshing.depth
MESH_TRIM = 7                         # trims low-density Poisson artefacts
MIN_MESH_COMPONENT_FRACTION = 0.02    # drop connected components smaller than this
                                       # fraction of the largest component (declutters
                                       # floating Poisson debris away from the target)

# ----------------------------------------------------------------------
# 8. MISC
# ----------------------------------------------------------------------
RANDOM_SEED = 42
