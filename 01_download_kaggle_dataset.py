# 01_download_kaggle_dataset.py
"""
Urban Place Perception - Human-Centered GeoAI
Pipeline Step 1 - Download & Prepare Dataset
=============================================
Downloads the Google Street View dataset from Kaggle and prepares a
reproducible sample for downstream inference.

This script replaces the Mapillary API approach (which suffered persistent
500 errors) with a fast Kaggle-based download (~600 MB zip, ~2-5 min).

Steps:
  1️⃣  Downloads the "paulchambaz/google-street-view" dataset via the Kaggle CLI.
  2️⃣  Extracts the zip into data/kaggle_gsv/.
  3️⃣  Parses coords.csv for latitude/longitude of every image.
  4️⃣  Randomly samples SAMPLE_SIZE images and copies them into data/images/.
  5️⃣  Writes data/metadata.csv (same schema the downstream pipeline expects).

Prerequisites:
  - pip install kaggle
  - Place your kaggle.json token at ~/.kaggle/kaggle.json
    (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)
    Get it from: https://www.kaggle.com/settings → "Create New Token"

Run with:  python 01_download_kaggle_dataset.py
"""

import os
import csv
import glob
import shutil
import zipfile
import pathlib
import random
import subprocess
import sys

from config import DATA_DIR, SAMPLE_SIZE, SEED

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KAGGLE_DATASET_SLUG = "paulchambaz/google-street-view"
KAGGLE_ZIP_NAME = "google-street-view.zip"
EXTRACT_DIR = os.path.join(DATA_DIR, "kaggle_gsv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")


# ---------------------------------------------------------------------------
# Step 1 - Download from Kaggle
# ---------------------------------------------------------------------------
def _download_from_kaggle():
    """Download the dataset zip using the kaggle CLI."""
    zip_path = os.path.join(DATA_DIR, KAGGLE_ZIP_NAME)
    if os.path.exists(zip_path):
        print(f"[OK] {KAGGLE_ZIP_NAME} already exists - skipping download.")
        return zip_path

    # Check that kaggle is importable (i.e. installed)
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[ERROR] The 'kaggle' package is not installed.")
        print("        Run:  pip install kaggle")
        print("        Then place your API token at ~/.kaggle/kaggle.json")
        sys.exit(1)

    print(f"[DL] Downloading {KAGGLE_DATASET_SLUG} from Kaggle ...")
    subprocess.check_call([
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", KAGGLE_DATASET_SLUG,
        "-p", DATA_DIR,
    ])

    # The kaggle CLI names the zip after the dataset slug
    # It may be named "google-street-view.zip" or just the slug tail
    if os.path.exists(zip_path):
        return zip_path

    # Fallback: look for any zip that was just created
    zips = sorted(glob.glob(os.path.join(DATA_DIR, "*.zip")),
                  key=os.path.getmtime, reverse=True)
    if zips:
        found = zips[0]
        print(f"[DL] Found downloaded archive: {found}")
        return found

    print("[ERROR] Kaggle download completed but no zip file found in", DATA_DIR)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2 - Extract
# ---------------------------------------------------------------------------
def _extract_zip(zip_path: str):
    """Extract the zip if the target directory is empty or missing."""
    if os.path.isdir(EXTRACT_DIR) and os.listdir(EXTRACT_DIR):
        print(f"[ZIP] {EXTRACT_DIR} already extracted - skipping.")
        return
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    print(f"[ZIP] Extracting {zip_path} -> {EXTRACT_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print("[ZIP] Extraction complete.")


# ---------------------------------------------------------------------------
# Step 3 - Parse coords.csv
# ---------------------------------------------------------------------------
def _parse_coords() -> dict:
    """Read coords.csv and return {image_filename: (lat, lon)}.
    Handles cases where the CSV may not have a header or may contain only latitude and longitude columns.
    If the required columns cannot be detected, returns an empty mapping and logs a warning.
    """
    coords_map = {}

    # Search for coords.csv in the extracted tree
    coords_file = None
    for root, _dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.lower() == "coords.csv":
                coords_file = os.path.join(root, f)
                break
        if coords_file:
            break

    if not coords_file:
        print("[ERROR] coords.csv not found inside the extracted dataset.")
        sys.exit(1)

    print(f"[MAP] Parsing coordinates from {coords_file} ...")
    with open(coords_file, newline="", encoding="utf-8") as cf:
        reader = csv.reader(cf)
        rows = list(reader)
        if not rows:
            print("[ERROR] coords.csv is empty.")
            sys.exit(1)

        # Determine if the first row is a header by checking if it contains non-numeric values
        first_row = rows[0]
        def is_float(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        header_present = not (all(is_float(val) for val in first_row[:2]))
        start_idx = 1 if header_present else 0

        # Default column indices assuming format: image, lat, lon OR lat, lon
        img_col = 0 if header_present else None
        lat_col = 1 if header_present else 0
        lon_col = 2 if header_present else 1

        # If header present, attempt to locate named columns
        if header_present:
            header = [c.strip().lower() for c in first_row]
            # Find lat/lon columns
            for i, name in enumerate(header):
                if name in ("latitude", "lat"):
                    lat_col = i
                elif name in ("longitude", "lng", "lon"):
                    lon_col = i
                elif name in ("image", "filename", "file", "id", "name", "img"):
                    img_col = i
            # If we could not locate lat/lon, fall back to positional defaults

        # Process data rows
        for row in rows[start_idx:]:
            if len(row) <= max(lat_col, lon_col):
                continue
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            except ValueError:
                continue
            # Determine image key if possible
            if img_col is not None and len(row) > img_col:
                img_name = row[img_col].strip()
            else:
                # No image identifier column; use the row index as the key.
                # This matches datasets where images are named 0.png, 1.png, …
                img_name = str(len(coords_map))
            coords_map[img_name] = (lat, lon)

    print(f"   Found coordinates for {len(coords_map)} entries.")
    return coords_map


# ---------------------------------------------------------------------------
# Step 4 - Find all images in the extracted folder
# ---------------------------------------------------------------------------
def _find_all_images() -> list:
    """Return a list of absolute paths to image files."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for root, _dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            if pathlib.Path(f).suffix.lower() in exts:
                images.append(os.path.join(root, f))
    images.sort()
    print(f"[IMG] Found {len(images)} image files in {EXTRACT_DIR}.")
    return images


# ---------------------------------------------------------------------------
# Step 5 - Sample, copy, and write metadata
# ---------------------------------------------------------------------------
def _sample_and_write(images: list, coords_map: dict):
    """Pick SAMPLE_SIZE images, copy to data/images/, write metadata.csv."""
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Build a list of (path, lat, lon) tuples - prefer images with coords
    paired = []
    for img_path in images:
        basename = os.path.basename(img_path)
        name_no_ext = pathlib.Path(basename).stem
        # Try matching by full name, then by stem, then by just the number
        lat, lon = coords_map.get(basename,
                    coords_map.get(name_no_ext,
                    coords_map.get(basename.lower(), (None, None))))
        if lat is not None and lon is not None:
            paired.append((img_path, lat, lon))

    if len(paired) < SAMPLE_SIZE:
        print(f"[WARN] Only {len(paired)} images have coordinates; "
              f"falling back to using all {len(images)} images.")
        # Use all images - set coord to (0, 0) for those without
        paired = []
        for img_path in images:
            basename = os.path.basename(img_path)
            name_no_ext = pathlib.Path(basename).stem
            lat, lon = coords_map.get(basename,
                        coords_map.get(name_no_ext,
                        coords_map.get(basename.lower(), (0.0, 0.0))))
            paired.append((img_path, lat, lon))

    # Reproducible random sample
    random.seed(SEED)
    selected = random.sample(paired, min(SAMPLE_SIZE, len(paired)))

    print(f"[OK] Selected {len(selected)} images (seed={SEED}).")

    rows = []
    for idx, (img_path, lat, lon) in enumerate(selected, start=1):
        ext = pathlib.Path(img_path).suffix
        dest_name = f"img_{idx:03d}{ext}"
        dest_path = os.path.join(IMAGES_DIR, dest_name)
        shutil.copy2(img_path, dest_path)
        rows.append({
            "image_id": f"img_{idx:03d}",
            "filepath": os.path.abspath(dest_path),
            "latitude": lat,
            "longitude": lon,
        })

    # Write metadata CSV
    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=["image_id", "filepath",
                                            "latitude", "longitude"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[META] Metadata written to {METADATA_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Download
    zip_path = _download_from_kaggle()

    # Step 2: Extract
    _extract_zip(zip_path)

    # Step 3: Parse coordinates
    coords_map = _parse_coords()

    # Step 4: Find images
    images = _find_all_images()
    if not images:
        print("[ERROR] No images found in extracted dataset.")
        sys.exit(1)

    # Step 5: Sample and write
    _sample_and_write(images, coords_map)

    # Summary
    n_images = sum(1 for _ in pathlib.Path(IMAGES_DIR).rglob("*.[jp][pn]g"))
    print("\n=== Preparation complete ===")
    print(f"Total images in data/images/: {n_images}")
    print(f"Metadata CSV: {METADATA_CSV}")


if __name__ == "__main__":
    main()
