# PixelOps — UAV Photogrammetry 3D Reconstruction Pipeline

An end-to-end, real-data photogrammetry pipeline: UAV video in, a genuine
dense point cloud + mesh + interactive 360° viewer out. No synthetic
geometry, no placeholder confidence, no toy data anywhere in the path.

This copy of the project is **dataset-agnostic** — everything specific to
one target lives in `src/config.py` (`TARGET_NAME`, `OUTPUT_BASENAME`) and
in `dataset/videos/` (your actual footage). Point it at any UAV video set.

---

## 1. Project objective

Reconstruct a real 3D scene from real drone video using classical
multi-view photogrammetry (SfM + MVS), and let you inspect the result
from any angle in a browser.

Pipeline:

```
Real UAV videos
  → inspect (fps/res/codec/duration — measured, never assumed)
  → adaptive frame extraction
  → sharpness/exposure/duplicate filtering
  → COLMAP feature extraction + matching
  → COLMAP sparse SfM (camera poses + sparse points)
  → COLMAP dense MVS (patch_match_stereo + stereo_fusion)   [needs GPU]
  → depth maps (COLMAP's real per-pixel stereo depth)
  → confidence maps (geometric vs photometric depth agreement)
  → dense colored point cloud (Open3D cleanup)
  → mesh (COLMAP poisson_mesher + Open3D cleanup)
  → interactive 360° three.js viewer
  → reconstruction_report.json with full diagnostics
```

## 2. Dataset placement

Drop your video files (`.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`) into:

```
Pixelops/dataset/videos/
```

Any number of clips, any resolution/fps/codec — Stage 1 measures each
file's real properties itself. Edit `TARGET_NAME` in `src/config.py` to
describe what you're reconstructing (used only in the report/viewer
title, not in processing).

## 3. Installation (Windows 11 / PowerShell)

```powershell
cd Pixelops
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. COLMAP installation

COLMAP is a separate binary, not a pip package.

**Option A — prebuilt Windows binary (simplest):**
1. Download the latest Windows release from
   `https://github.com/colmap/colmap/releases` (a `COLMAP-x.y.z-windows-cuda.zip`
   for GPU support, or the non-CUDA zip for CPU-only sparse SfM).
2. Unzip anywhere, e.g. `C:\colmap`.
3. Add `C:\colmap` to your `PATH` (System Properties → Environment
   Variables), so the command `colmap` resolves from any shell.
4. Verify:
   ```powershell
   colmap -h
   ```

**Option B — via conda:**
```powershell
conda install -c conda-forge colmap
```

**GPU note:** Dense MVS (`patch_match_stereo`) requires a CUDA build of
COLMAP + an NVIDIA GPU. The pipeline auto-detects GPU availability
(`nvidia-smi`) and, if none is found, automatically skips dense
MVS/depth/confidence/mesh — it still delivers a genuine sparse point
cloud and viewer rather than failing outright or faking dense output.
Use `run.bat --skip-dense` to force sparse-only even with a GPU present.

## 5. Configuration

Everything tunable lives in `src/config.py`:
- `TARGET_NAME`, `OUTPUT_BASENAME` — identify this dataset/run
- `TARGET_CANDIDATE_FRAMES_PER_VIDEO`, `MIN_FRAME_INTERVAL_SECONDS` — extraction density
- `MIN_SHARPNESS_VARIANCE`, brightness bounds — quality gate
- `MATCHER_TYPE` (`sequential` for video, `exhaustive` for unordered photos)
- `POISSON_DEPTH`, outlier-removal parameters, etc.

Every threshold is commented with what it controls and why.

## 6. One-command execution

```powershell
.\.venv\Scripts\python.exe src\main.py
```
or
```powershell
run.bat
```

Add `--skip-dense` to force a sparse-only run (useful for a quick sanity
check before committing to a long dense-MVS pass).

## 7. Expected outputs

```
outputs/
├── inspection/video_inventory.json
├── frames/<video>/frame_XXXXXX.jpg + frame_manifest.json
├── quality/quality_report.json
├── sparse/model_txt/, sparse_points.ply
├── depth/<image>_depth.png (16-bit), <image>_depth_colorized.png
├── confidence/<image>_confidence.png, confidence_summary.json
├── pointcloud/<name>_pointcloud.ply, pointcloud_stats.json
├── mesh/<name>_mesh.ply, mesh_stats.json
├── visualization/pointcloud_view.png, mesh_view.png
├── viewer/<name>_360.html + copy of the model file it loads
└── reconstruction_report.json   ← full run summary + validation
```

## 8. How to open the 360° viewer

Browsers block loading local files via `fetch()` from a `file://` URL, so
serve the folder over plain HTTP:

```powershell
cd outputs\viewer
python -m http.server 8000
```
Then open `http://localhost:8000/<name>_360.html` in a browser.
Drag = rotate, scroll = zoom, right-drag = pan. The camera auto-frames
around whatever geometry actually loaded.

**This is an interactive 3D viewer of the reconstructed geometry, not an
equirectangular 360° photo panorama.** A true panorama requires a single
viewpoint with full angular coverage around it, which UAV flight-path
footage generally does not provide (the camera keeps moving/translating
through the scene). If your footage genuinely contains a stationary
360° sweep, you can add a dedicated panorama-stitching stage (e.g. with
OpenCV's `Stitcher` class) — this project deliberately does not fabricate
one from unrelated frames.

## 9. How to interpret depth

Each `<image>_depth.png` is COLMAP's real **geometric-consistency depth
map** from `patch_match_stereo` — one independent depth estimate per
pixel, agreed on across multiple neighbouring camera views, not a single
photo's blur/defocus cue. Values are 16-bit, scaled per-image between
that image's own min/max valid depth (not metric — see scale note
below). The `_colorized.png` twin is a Turbo-colormap visualization for
quick inspection; black pixels = no valid depth (occluded, textureless,
or outside all overlapping views).

## 10. How to interpret confidence

Confidence is **not** depth re-colored and **not** random. For every
pixel:

```
confidence = clamp(1 − |depth_geometric − depth_photometric| / depth_geometric, 0, 1)
```

`depth_geometric` and `depth_photometric` are COLMAP's two independent
stereo estimates for that pixel. Where they agree closely, the surface
point is well-observed from multiple viewpoints → high confidence
(bright). Where they disagree, or geometric stereo found no valid depth
at all, confidence is low or zero (dark). See
`outputs/confidence/confidence_summary.json` for the exact method string
and per-image mean confidence.

## 11. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `COLMAP executable not found on PATH` | Install COLMAP (§4) and confirm `colmap -h` works in a fresh terminal. |
| Mapper produces multiple disconnected components | Footage has a gap in visual overlap (drone left and re-entered the scene, or videos don't share viewpoints). Check `reconstruction_report.json → sfm`; consider trimming/reordering clips or increasing `SEQUENTIAL_OVERLAP`. |
| `Only N images registered` error | Too few usable frames or insufficient overlap/texture. Inspect `outputs/quality/quality_report.json`'s sharpness distribution; lower `MIN_SHARPNESS_VARIANCE` if it rejected too aggressively, or re-fly with more overlap. |
| Dense stage skipped / "no_gpu" | No CUDA-capable COLMAP + NVIDIA GPU detected. Sparse point cloud + viewer are still produced. Install a CUDA build of COLMAP on a machine with an NVIDIA GPU for dense/mesh output. |
| Viewer shows nothing / console CORS error | You opened the HTML via `file://`. Serve it (§8) with `python -m http.server`. |
| `stereo_fusion produced an empty fused.ply` | Insufficient geometric consistency between views — usually not enough overlapping frames survived filtering. Loosen `MIN_SHARPNESS_VARIANCE` slightly or increase `TARGET_CANDIDATE_FRAMES_PER_VIDEO`. |
| Re-running seems to mix old and new results | It shouldn't — `workspace/images`, `workspace/sparse`, `workspace/dense`, and `outputs/frames` are wiped and rebuilt at the start of the relevant stage on every run. If you changed `dataset/videos/` contents, just re-run. |

## 12. Photogrammetry limitations (read before presenting results)

- **No absolute scale.** COLMAP reconstructs geometry up to an arbitrary
  scale factor unless you supply ground-control points (GCPs) or a known
  real-world measurement. `pointcloud_stats.json` and the final report
  explicitly flag this. To add metric scale: measure one known real
  distance in the scene, measure the same two points' distance in the
  output point cloud, and multiply all coordinates by
  `real_distance / reconstructed_distance` (a short Open3D script can
  apply this scale transform to the final PLY).
- **Coverage gaps are real, not bugs.** Any surface never seen by the
  drone (backs of objects, ground directly under the flight path,
  areas outside all camera frustums) will simply be absent from the
  point cloud/mesh — the pipeline does not interpolate or hallucinate
  unseen geometry beyond what Poisson surface reconstruction
  mathematically infers between *observed* nearby points.
- **Dense MVS needs a GPU.** This is a COLMAP/CUDA requirement, not a
  limitation of this codebase; sparse SfM alone still runs on CPU.

## 13. Final checklist — is this a REAL reconstruction of my footage?

- [ ] `outputs/inspection/video_inventory.json` lists YOUR actual video filenames, resolutions and durations (not placeholders)
- [ ] `outputs/reconstruction_report.json → sfm.num_registered_images` is close to your accepted frame count, not a handful
- [ ] `outputs/sparse/sparse_points.ply` opens in any PLY viewer and visibly resembles your scene's rough layout
- [ ] `outputs/confidence/confidence_summary.json → confidence_method` matches the formula in §10 (not "random" or "= depth")
- [ ] `outputs/mesh/<name>_mesh.ply` (if dense MVS ran) has vertex/triangle counts > 0 in `mesh_stats.json`
- [ ] The 360° viewer, once served over HTTP, visibly loads and lets you orbit YOUR reconstructed shape — not a generic sample model
- [ ] `reconstruction_report.json → warnings` has been read and understood (empty is ideal, but honest warnings about skipped stages are expected on CPU-only machines)
