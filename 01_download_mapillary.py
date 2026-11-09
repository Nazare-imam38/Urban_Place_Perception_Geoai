# 01_download_mapillary.py
"""
Download real NYC street‑view images from Mapillary and generate metadata.

This version adds:
- Pagination support for each tile (so we get *all* images up to the desired count).
- Robust retry logic for tile queries and image downloads (exponential back‑off).
- Fallback to the lower‑resolution thumbnail when the high‑resolution URL is missing.
- Early‑exit once the required number of images (`SAMPLE_SIZE`) is collected, cutting down the number of API calls dramatically.
- Clear console messages that work on Windows terminals.
"""

import os
import csv
import pathlib
import random
import time
from typing import List, Dict
import requests
from tqdm import tqdm

from config import (
    MAPILLARY_ACCESS_TOKEN,
    MAPILLARY_BBOX,
    SAMPLE_SIZE,
    SEED,
    DATA_DIR,
)

BASE_URL = "https://graph.mapillary.com/images"
# Prefer the high‑resolution thumbnail; fall back to the original size if unavailable.
FIELDS = "id,computed_geometry,thumb_2048_url,thumb_original_url"

# ----------------------------------------------------------------------
# Helper: perform a GET request with exponential back‑off
# ----------------------------------------------------------------------
def _http_get(url: str, params: Dict, max_retries: int = 3) -> requests.Response | None:
    backoff = 2  # seconds
    headers = {"User-Agent": "UrbanPlacePerception/1.0"}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == max_retries:
                print(f"[ERROR] Request to {url} failed after {max_retries} attempts: {e}")
                return None
            else:
                print(f"[WARN] Attempt {attempt} failed ({e}); retrying in {backoff}s …")
                time.sleep(backoff)
                backoff *= 2
    return None

# ----------------------------------------------------------------------
# Query a single tile, handling pagination until we have enough images.
# ----------------------------------------------------------------------
def _query_tile(cell_bbox: str, needed: int, already: int) -> List[Dict]:
    images: List[Dict] = []
    params = {
        "access_token": MAPILLARY_ACCESS_TOKEN,
        "bbox": cell_bbox,
        "fields": FIELDS,
        "limit": 500,
    }
    while True:
        resp = _http_get(BASE_URL, params)
        if resp is None:
            break  # give up on this tile
        data = resp.json()
        for img in data.get("data", []):
            geom = img.get("computed_geometry", {})
            lat_i = geom.get("lat")
            lon_i = geom.get("lon")
            thumb = img.get("thumb_2048_url") or img.get("thumb_original_url")
            if lat_i is not None and lon_i is not None and thumb:
                images.append({"id": img["id"], "lat": lat_i, "lon": lon_i, "thumb_url": thumb})
                if len(images) + already >= needed:
                    return images
        # Pagination: Mapillary returns a "next" link in "links" when more pages exist.
        next_link = data.get("links", {}).get("next")
        if not next_link:
            break
        # The next link already contains all query parameters; request it directly.
        resp = _http_get(next_link, {})
        if resp is None:
            break
        # Loop continues with the new response.
    return images

# ----------------------------------------------------------------------
# Gather images across the whole NYC bbox, stopping early when SAMPLE_SIZE is met.
# ----------------------------------------------------------------------
def _query_mapillary(bbox: str, desired: int = SAMPLE_SIZE) -> List[Dict]:
    west, south, east, north = map(float, bbox.split(","))
    step = 0.009  # safely under the 0.01 deg API limit per request

    # Compute total tile count for a nice progress bar.
    lat_steps = int((north - south) / step) + 1
    lon_steps = int((east - west) / step) + 1
    total_tiles = lat_steps * lon_steps

    tile_bar = tqdm(total=total_tiles, desc="[🧭] Querying tiles", unit="tile")
    collected: List[Dict] = []
    stop = False

    lat = south
    while lat < north and not stop:
        lon = west
        while lon < east and not stop:
            cell_bbox = (
                f"{lon:.6f},{lat:.6f},{min(lon + step, east):.6f},{min(lat + step, north):.6f}"
            )
            needed = desired - len(collected)
            tile_images = _query_tile(cell_bbox, needed, len(collected))
            collected.extend(tile_images)
            if len(collected) >= desired:
                stop = True
            lon += step
            tile_bar.update(1)
        lat += step
    tile_bar.close()
    return collected

# ----------------------------------------------------------------------
# Download a single image with retry and a progress bar.
# ----------------------------------------------------------------------
def _download_image(url: str, dest_path: pathlib.Path, max_retries: int = 3) -> bool:
    backoff = 2
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest_path.name,
                leave=False,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            if attempt == max_retries:
                print(f"[ERROR] Failed to download {url} after {max_retries} attempts: {e}")
                return False
            else:
                print(f"[WARN] Image download attempt {attempt} failed ({e}); retrying in {backoff}s …")
                time.sleep(backoff)
                backoff *= 2
    return False

# ----------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------
def main():
    images_dir = pathlib.Path(DATA_DIR) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Querying Mapillary for NYC images …")
    all_images = _query_mapillary(MAPILLARY_BBOX, desired=SAMPLE_SIZE)
    if len(all_images) < SAMPLE_SIZE:
        raise RuntimeError(
            f"Mapillary returned only {len(all_images)} images; need at least {SAMPLE_SIZE}."
        )

    random.seed(SEED)
    selected = random.sample(all_images, SAMPLE_SIZE)
    print(f"[✅] Selected {SAMPLE_SIZE} images (seed={SEED}).")

    for idx, img in enumerate(tqdm(selected, desc="[⬇️] Downloading images", unit="image"), start=1):
        out_path = images_dir / f"img_{idx:03d}.jpg"
        success = _download_image(img["thumb_url"], out_path)
        if success:
            img["local_path"] = str(out_path)
            print(f"[🖼️] Saved img_{idx:03d}.jpg (lat={img['lat']:.5f}, lon={img['lon']:.5f})")
        else:
            print(f"[⚠️] Skipped img_{idx:03d} due to download failure.")

    metadata_path = pathlib.Path(DATA_DIR) / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "filepath", "latitude", "longitude"])
        for idx, img in enumerate(selected, start=1):
            writer.writerow([
                f"img_{idx:03d}",
                os.path.abspath(img.get("local_path", "")),
                img["lat"],
                img["lon"],
            ])
    print("\n=== Preparation complete ===")
    print(f"Total images downloaded: {SAMPLE_SIZE}")
    print(f"Metadata CSV: {metadata_path}")

if __name__ == "__main__":
    main()
