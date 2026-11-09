# 06_spatial_summary.py
"""
Urban Place Perception - Human-Centered GeoAI
Pipeline Step 6 - Spatial Aggregation & Summary
================================================
Generates a grid-based spatial aggregation of perception scores, enabling
neighbourhood-level analysis of urban place perception patterns.

Outputs:
    - outputs/vlm/spatial_summary.geojson   (grid polygons with mean scores)
    - outputs/vlm/spatial_summary.csv       (same data in tabular form)

The grid cell size (in metres) is controlled via ``config.GRID_SIZE_METERS``.
If geopandas is not installed, the script falls back to a simple CSV summary.

Run with:  python 06_spatial_summary.py
"""

import pandas as pd
from pathlib import Path

# Optional: geopandas adds heavy dependency, but it gives real geometry handling.
# If geopandas is not installed the script will fall back to a simple CSV summary.
try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    GEOPANDAS_AVAILABLE = True
except Exception:      # pragma: no cover
    GEOPANDAS_AVAILABLE = False

# Maximum number of grid cells to generate (to avoid MemoryError)
MAX_GRID_CELLS = 5000  # Adjust as needed for larger datasets

from config import OUTPUT_DIR, GRID_SIZE_METERS


def _load_results() -> pd.DataFrame:
    """Read the inference CSV and drop rows without valid coordinates."""
    results_path = Path(OUTPUT_DIR) / "results.csv"
    df = pd.read_csv(results_path)
    # Remove placeholder (0,0) coordinates
    df = df[~((df["latitude"] == 0) & (df["longitude"] == 0))]
    return df


def _make_geodataframe(df: pd.DataFrame) -> "gpd.GeoDataFrame":
    """Turn the latitude/longitude columns into a GeoDataFrame."""
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    # Project to a metric CRS for accurate grid sizing (Web Mercator works globally)
    gdf = gdf.to_crs(epsg=3857)
    return gdf


def _create_grid(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Create a rectangular grid that covers the extent of the points.
    Returns None if the grid would be excessively large.
    """
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    xmin, ymin, xmax, ymax = bounds
    cell_size = GRID_SIZE_METERS

    # Generate grid cell coordinates
    cols = list(range(int((xmin - xmin % cell_size)), int(xmax + cell_size), int(cell_size)))
    rows = list(range(int((ymin - ymin % cell_size)), int(ymax + cell_size), int(cell_size)))

    # Check for excessive number of cells
    if len(cols) * len(rows) > MAX_GRID_CELLS:
        print(f"[WARNING] Grid would contain {len(cols) * len(rows)} cells (> {MAX_GRID_CELLS}). Skipping grid creation to avoid MemoryError.")
        return None

    polygons = []
    for x in cols:
        for y in rows:
            polygons.append(box(x, y, x + cell_size, y + cell_size))

    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=gdf.crs)
    return grid


def _aggregate_to_grid(gdf: "gpd.GeoDataFrame", grid: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Spatial join points to grid cells and compute mean scores per cell."""
    joined = gpd.sjoin(gdf, grid, how="inner", predicate="within")
    agg_funcs = {
        "safety_score": "mean",
        "walkability_score": "mean",
        "greenery_score": "mean",
        "visual_quality_score": "mean",
        "confidence": lambda x: "high" if (x == "high").sum() >= len(x) / 2 else "low",
    }
    grid_stats = joined.groupby("index_right").agg(agg_funcs).reset_index()
    result = grid.merge(grid_stats, left_index=True, right_on="index_right", how="left")
    result = result.drop(columns=["index_right"])
    return result


def main() -> None:
    df = _load_results()
    if df.empty:
        print("[WARNING] No valid records found in results.csv")
        return

    if not GEOPANDAS_AVAILABLE:
        # Fallback to simple CSV aggregation (already implemented)
        summary = (
            df.groupby(["latitude", "longitude"])
            .agg(
                safety_score=("safety_score", "mean"),
                walkability_score=("walkability_score", "mean"),
                greenery_score=("greenery_score", "mean"),
                visual_quality_score=("visual_quality_score", "mean"),
                confidence=("confidence", lambda x: "high" if (x == "high").sum() >= len(x) / 2 else "low"),
            )
            .reset_index()
        )
        out_path = Path(OUTPUT_DIR) / "spatial_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"[OK] Spatial CSV summary saved -> {out_path}")
        return

    gdf = _make_geodataframe(df)
    grid = _create_grid(gdf)
    if grid is None:
        # Grid creation skipped; fall back to CSV summary
        summary = (
            df.groupby(["latitude", "longitude"])
            .agg(
                safety_score=("safety_score", "mean"),
                walkability_score=("walkability_score", "mean"),
                greenery_score=("greenery_score", "mean"),
                visual_quality_score=("visual_quality_score", "mean"),
                confidence=("confidence", lambda x: "high" if (x == "high").sum() >= len(x) / 2 else "low"),
            )
            .reset_index()
        )
        out_path = Path(OUTPUT_DIR) / "spatial_summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"[OK] Spatial CSV summary saved (grid skipped) -> {out_path}")
        return

    aggregated = _aggregate_to_grid(gdf, grid)

    out_geojson = Path(OUTPUT_DIR) / "spatial_summary.geojson"
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_file(out_geojson, driver="GeoJSON")
    print(f"[OK] Spatial GeoJSON saved -> {out_geojson}")

    out_csv = Path(OUTPUT_DIR) / "spatial_summary.csv"
    aggregated.drop(columns="geometry").to_csv(out_csv, index=False)
    print(f"[OK] Spatial CSV saved -> {out_csv}")

if __name__ == "__main__":
    main()
