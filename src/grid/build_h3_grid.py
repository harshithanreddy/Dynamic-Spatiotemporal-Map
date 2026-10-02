"""
Step 1: Build an H3 hexagonal grid covering the specified region land area.

Output: data/processed/<region_name>/hex_grid.parquet
  columns: hex_id (str), geometry (shapely Polygon, WGS84), lon (centroid), lat (centroid)

Also writes data/processed/<region_name>/hex_neighbors.json: {hex_id: [neighbor hex_ids]}
for graph edge construction later.
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Polygon

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import GEOLOGY_POLY, H3_RESOLUTION, PROCESSED, WGS84, RegionConfig, get_region


def get_land_boundary(region: RegionConfig) -> gpd.GeoDataFrame:
    """
    Prefer clipping to a real land polygon so hexes aren't generated over open water.
    Falls back to the region bbox if no boundary source is available.
    """
    bbox = region.bbox

    if GEOLOGY_POLY.exists():
        gdf = gpd.read_file(GEOLOGY_POLY).to_crs(WGS84)
        gdf = gdf.cx[
            bbox["min_lon"] : bbox["max_lon"],
            bbox["min_lat"] : bbox["max_lat"],
        ]
        if not gdf.empty:
            boundary = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
            print(f"[grid] Using geology polygon boundary for region '{region.name}' (bounds: {boundary.bounds})")
            return gpd.GeoDataFrame(geometry=[boundary], crs=WGS84)

    bbox_poly = Polygon(
        [
            (bbox["min_lon"], bbox["min_lat"]),
            (bbox["max_lon"], bbox["min_lat"]),
            (bbox["max_lon"], bbox["max_lat"]),
            (bbox["min_lon"], bbox["max_lat"]),
        ]
    )
    print(f"[grid] WARNING: falling back to bbox for region '{region.name}' (bounds: {bbox_poly.bounds})")
    return gpd.GeoDataFrame(geometry=[bbox_poly], crs=WGS84)


def geom_to_h3_cells(geom, resolution):
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    cells = set()
    for poly in polys:
        outer = [(lat, lng) for lng, lat in poly.exterior.coords]
        holes = [[(lat, lng) for lng, lat in interior.coords] for interior in poly.interiors]
        h3poly = h3.LatLngPoly(outer, *holes)
        cells.update(h3.polygon_to_cells(h3poly, resolution))
    return cells


def build_grid(region: RegionConfig, resolution: int = H3_RESOLUTION):
    land = get_land_boundary(region)
    land_geom = land.geometry.iloc[0]

    hexes = geom_to_h3_cells(land_geom, resolution)

    rows = []
    for hex_id in hexes:
        boundary = h3.cell_to_boundary(hex_id)
        poly = Polygon([(lng, lat) for lat, lng in boundary])
        lat, lng = h3.cell_to_latlng(hex_id)
        rows.append({"hex_id": hex_id, "geometry": poly, "lon": lng, "lat": lat})

    gdf = gpd.GeoDataFrame(rows, crs=WGS84)
    print(f"[grid] Built {len(gdf)} hexes at H3 resolution {resolution} for region '{region.name}'")
    return gdf, land


def build_neighbor_map(gdf: gpd.GeoDataFrame) -> dict:
    hex_ids = set(gdf["hex_id"])
    neighbors = {}
    for hex_id in hex_ids:
        ring = set(h3.grid_disk(hex_id, 1)) - {hex_id}
        neighbors[hex_id] = sorted(ring & hex_ids)  # keep only neighbors inside the study area
    return neighbors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build H3 grid for a target region.")
    parser.add_argument("--region", type=str, choices=["noto", "hokkaido"], default="noto", help="Region name (e.g. noto, hokkaido)")
    args = parser.parse_args()

    region = get_region(args.region)
    processed_dir = PROCESSED / region.name
    processed_dir.mkdir(parents=True, exist_ok=True)

    grid, land = build_grid(region)
    grid_out = processed_dir / "hex_grid.parquet"
    grid.to_parquet(grid_out)
    print(f"[grid] Saved -> {grid_out}")

    neighbors = build_neighbor_map(grid)
    neighbors_out = processed_dir / "hex_neighbors.json"
    with open(neighbors_out, "w") as f:
        json.dump(neighbors, f)
    avg_degree = sum(len(v) for v in neighbors.values()) / len(neighbors)
    print(f"[grid] Saved neighbor map -> {neighbors_out} (avg degree {avg_degree:.1f})")

    land_bounds = land.geometry.iloc[0].bounds
    grid_bounds = tuple(grid.total_bounds)
    print(f"[grid] Total hex count: {len(grid)}")
    print(f"[grid] Grid bounds (minx, miny, maxx, maxy): {grid_bounds}")
    print(f"[grid] Land boundary bounds (minx, miny, maxx, maxy): {land_bounds}")
