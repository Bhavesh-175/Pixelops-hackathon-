from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[1]

FRONTEND = ROOT / "frontend"

OUTPUTS = ROOT / "outputs"


app = FastAPI(
    title="PixelOps UAV Photogrammetry"
)


@app.get("/api/status")
def status():

    files = {}

    for filename in [
        "reconstruction.ply",
        "dense_cloud.ply",
        "mesh.ply",
        "model.obj",
        "model.glb"
    ]:

        path = OUTPUTS / filename

        files[filename] = {
            "exists": path.exists(),
            "size": path.stat().st_size
            if path.exists()
            else 0
        }

    return {
        "project": "PixelOps",
        "files": files
    }


@app.get("/api/file/{filename}")
def get_file(filename: str):

    path = OUTPUTS / filename

    if not path.exists():
        return {
            "error": "File not found"
        }

    return {
        "filename": filename,
        "size": path.stat().st_size
    }


app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND,
        html=True
    ),
    name="frontend"
)