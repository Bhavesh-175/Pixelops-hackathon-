from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]

COLMAP_DIR = ROOT / "data" / "cache" / "colmap"

SPARSE_DIR = COLMAP_DIR / "sparse"
DENSE_DIR = COLMAP_DIR / "dense"

OUTPUT_DIR = ROOT / "outputs"


def run_command(command):

    print("\nRunning:")
    print(" ".join(map(str, command)))

    subprocess.run(
        command,
        check=True
    )


def run_dense_reconstruction():

    sparse_model = SPARSE_DIR / "0"

    DENSE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    run_command([
        "colmap",
        "image_undistorter",
        "--image_path",
        str(ROOT / "data" / "frames"),
        "--input_path",
        str(sparse_model),
        "--output_path",
        str(DENSE_DIR),
        "--output_type",
        "COLMAP"
    ])

    run_command([
        "colmap",
        "patch_match_stereo",
        "--workspace_path",
        str(DENSE_DIR),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true"
    ])

    run_command([
        "colmap",
        "stereo_fusion",
        "--workspace_path",
        str(DENSE_DIR),
        "--workspace_format",
        "COLMAP",
        "--output_path",
        str(OUTPUT_DIR / "dense_cloud.ply")
    ])


if __name__ == "__main__":

    run_dense_reconstruction()