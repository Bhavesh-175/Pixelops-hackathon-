from pathlib import Path
import json


def save_sfm_status(
    registered_cameras,
    sparse_points,
    matches
):

    root = Path(__file__).resolve().parents[2]

    cache = root / "data" / "cache"

    cache.mkdir(
        parents=True,
        exist_ok=True
    )

    status = {
        "registered_cameras": registered_cameras,
        "sparse_points": sparse_points,
        "feature_matches": matches
    }

    with open(
        cache / "sfm_status.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            status,
            file,
            indent=4
        )


if __name__ == "__main__":

    print(
        "Full multi-view SfM is handled by COLMAP."
    )