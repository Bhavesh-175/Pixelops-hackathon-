from pathlib import Path
import cv2
import numpy as np


def analyze_image(image_path):

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    return {
        "filename": image_path.name,
        "width": width,
        "height": height,
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
    }


if __name__ == "__main__":

    root = Path(__file__).resolve().parents[2]

    frames = root / "data" / "frames"

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff",
    ]

    images = []

    for extension in extensions:
        images.extend(frames.glob(extension))

    for image in sorted(images):

        result = analyze_image(image)

        print(
            f"{result['filename']:25} "
            f"{result['width']}x{result['height']} "
            f"sharpness={result['sharpness']:.2f} "
            f"brightness={result['brightness']:.2f} "
            f"contrast={result['contrast']:.2f}"
        )