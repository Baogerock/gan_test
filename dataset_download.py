import shutil
import urllib.request
import zipfile
from pathlib import Path


def main() -> None:
    # Ensure dataset is placed under current project folder: ./pokemon
    target_dir = Path(__file__).resolve().parent / "pokemon"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Use a public Pokemon image source with no extra Python dependency.
    zip_url = "https://github.com/PokeAPI/sprites/archive/refs/heads/master.zip"
    zip_path = target_dir / "sprites_master.zip"
    extract_root = target_dir / "_tmp_extract"
    artwork_source = (
        extract_root
        / "sprites-master"
        / "sprites"
        / "pokemon"
        / "other"
        / "official-artwork"
    )

    print("Downloading Pokemon images to:", target_dir)
    urllib.request.urlretrieve(zip_url, zip_path)

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    copied = 0
    for img in artwork_source.glob("*.png"):
        dst = target_dir / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
            copied += 1

    print(f"Downloaded archive: {zip_path}")
    print(f"Copied {copied} images into: {target_dir}")

    # Cleanup temporary extraction folder and archive to save disk.
    shutil.rmtree(extract_root, ignore_errors=True)
    if zip_path.exists():
        zip_path.unlink()


if __name__ == "__main__":
    main()
