"""
viewer.py
=========
STAGE 10 - Interactive 360-degree viewer.

Generates a self-contained HTML file that loads three.js from a CDN and
reads the ACTUAL reconstructed mesh (preferred) or point cloud PLY file
that this run produced - never a placeholder. OrbitControls give full
rotate / zoom / pan around the model, auto-framed to its bounding box.

IMPORTANT: browsers block `fetch()` of local files from a file:// URL
(CORS), so the PLY must be served over HTTP. README explains the one
command needed (`python -m http.server`) to view it.

This is a genuine interactive 3D inspection tool, distinct from an
equirectangular 360 photo panorama - see panorama note in the report.

Output: outputs/viewer/<basename>_viewer.html
"""

from __future__ import annotations
from pathlib import Path

import config
from utils import log


VIEWER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
  html, body {{ margin:0; height:100%; background:#0b0e14; overflow:hidden; font-family: system-ui, sans-serif; }}
  #canvas-holder {{ width:100%; height:100%; }}
  #hud {{
    position:absolute; top:12px; left:12px; color:#d7dde5; background:rgba(10,14,20,0.65);
    padding:10px 14px; border-radius:8px; font-size:13px; line-height:1.5; max-width:360px;
  }}
  #hud b {{ color:#7fd1ff; }}
  #status {{ position:absolute; bottom:12px; left:12px; color:#9aa5b1; font-size:12px; }}
</style>
</head>
<body>
<div id="canvas-holder"></div>
<div id="hud">
  <b>{title}</b><br/>
  Source: <code>{model_filename}</code><br/>
  Drag = rotate &nbsp;|&nbsp; Scroll = zoom &nbsp;|&nbsp; Right-drag = pan
</div>
<div id="status">Loading real reconstructed geometry...</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/examples/js/loaders/PLYLoader.js"></script>
<script>
const holder = document.getElementById('canvas-holder');
const statusEl = document.getElementById('status');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e14);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.001, 10000);
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
holder.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 0.6);
dir.position.set(1, 1, 1);
scene.add(dir);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.screenSpacePanning = true;

const loader = new THREE.PLYLoader();
loader.load(
  "{model_relpath}",
  function (geometry) {{
    geometry.computeVertexNormals();
    let obj;
    const hasFaces = geometry.index !== null && geometry.index.count > 0;
    if (hasFaces) {{
      const material = new THREE.MeshStandardMaterial({{
        vertexColors: geometry.hasAttribute('color'),
        color: geometry.hasAttribute('color') ? 0xffffff : 0x8fb8ff,
        flatShading: false,
        side: THREE.DoubleSide,
      }});
      obj = new THREE.Mesh(geometry, material);
    }} else {{
      const material = new THREE.PointsMaterial({{
        size: 0.01,
        vertexColors: geometry.hasAttribute('color'),
        color: geometry.hasAttribute('color') ? 0xffffff : 0x8fb8ff,
      }});
      obj = new THREE.Points(geometry, material);
    }}
    scene.add(obj);

    // Auto-frame the camera around the ACTUAL loaded geometry bounding box
    geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere;
    const center = sphere.center;
    const radius = Math.max(sphere.radius, 0.001);

    controls.target.copy(center);
    camera.position.set(center.x + radius * 1.8, center.y + radius * 1.2, center.z + radius * 1.8);
    camera.near = radius / 100;
    camera.far = radius * 100;
    camera.updateProjectionMatrix();
    controls.update();

    statusEl.textContent = "Loaded " + (hasFaces ? "mesh" : "point cloud") +
      " - " + geometry.attributes.position.count.toLocaleString() + " vertices";
  }},
  function (xhr) {{
    if (xhr.total) {{
      statusEl.textContent = "Loading... " + Math.round((xhr.loaded / xhr.total) * 100) + "%";
    }}
  }},
  function (err) {{
    statusEl.textContent = "FAILED to load model file. See console + README (must be served over HTTP, not file://).";
    console.error(err);
  }}
);

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""


def run(mesh_info: dict, pointcloud_info: dict) -> Path:
    if mesh_info.get("mesh_available"):
        model_path = Path(mesh_info["output_mesh"])
    else:
        model_path = Path(pointcloud_info["output_ply"])
        log.info("Viewer will load the point cloud (no mesh was generated this run).")

    # Copy the model next to the viewer HTML so a single `python -m http.server`
    # from outputs/viewer/ serves both with a simple relative path.
    viewer_local_copy = config.VIEWER_DIR / model_path.name
    viewer_local_copy.write_bytes(model_path.read_bytes())

    html = VIEWER_TEMPLATE.format(
        title=f"{config.TARGET_NAME} - 360 Reconstruction Viewer",
        model_filename=model_path.name,
        model_relpath=model_path.name,
    )
    out_html = config.VIEWER_DIR / f"{config.OUTPUT_BASENAME}_360.html"
    out_html.write_text(html, encoding="utf-8")
    log.info(f"Wrote interactive viewer: {out_html}")
    log.info(f"Serve it with: cd {config.VIEWER_DIR} && python -m http.server 8000")
    return out_html
