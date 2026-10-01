from pathlib import Path
import cv2
import numpy as np


def create_sift():

    return cv2.SIFT_create(
        nfeatures=6000
    )


def extract_features(image_path):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    sift = create_sift()

    keypoints, descriptors = sift.detectAndCompute(
        image,
        None
    )

    if keypoints is None:
        keypoints = []

    if descriptors is None:
        descriptors = np.empty(
            (0, 128),
            dtype=np.float32
        )

    points = np.array(
        [kp.pt for kp in keypoints],
        dtype=np.float32
    )

    return points, descriptors


if __name__ == "__main__":

    root = Path(__file__).resolve().parents[2]

    frames = root / "data" / "frames"

    images = sorted(
        list(frames.glob("*.png")) +
        list(frames.glob("*.jpg")) +
        list(frames.glob("*.jpeg"))
    )

    for image in images:

        points, descriptors = extract_features(
            image
        )

        print(
            f"{image.name}: "
            f"{len(points)} features"
        )