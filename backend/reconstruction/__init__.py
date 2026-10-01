from pathlib import Path
import zipfile
import shutil


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "input"
FRAMES_DIR = ROOT / "data" / "frames"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def find_zip():
    zips = list(INPUT_DIR.glob("*.zip"))

    if not zips:
        raise FileNotFoundError(
            f"No ZIP file found in {INPUT_DIR}"
        )

    return zips[0]


def extract_images():
    zip_path = find_zip()

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old extracted images
    for item in FRAMES_DIR.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    count = 0

    with zipfile.ZipFile(zip_path, "r") as archive:

        for member in archive.infolist():

            if member.is_dir():
                continue

            suffix = Path(member.filename).suffix.lower()

            if suffix not in IMAGE_EXTENSIONS:
                continue

            filename = Path(member.filename).name

            if not filename:
                continue

            destination = FRAMES_DIR / filename

            with archive.open(member) as source:
                with open(destination, "wb") as target:
                    shutil.copyfileobj(source, target)

            count += 1

    print(f"ZIP: {zip_path.name}")
    print(f"Extracted images: {count}")
    print(f"Frames directory: {FRAMES_DIR}")

    return count


if __name__ == "__main__":
    extract_images()