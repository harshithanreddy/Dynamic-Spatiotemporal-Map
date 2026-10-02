"""
Step 4: Distance (meters) from each hex centroid to the nearest active fault.

Output: data/processed/<region_name>/fault_features.parquet
  columns: hex_id, fault_distance_m
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import METRIC_CRS, PROCESSED, WGS84, RegionConfig, get_region


def build_fault_features(region: RegionConfig) -> pd.DataFrame:
    """
    Computes distance (meters) from each hex centroid to the nearest active fault line.
    Supports both stub mode (0.0 distance), CSV point trace, and GeoJSON/GPKG vector fault lines.
    """
    processed_dir = PROCESSED / region.name
    processed_dir.mkdir(parents=True, exist_ok=True)

    hex_gdf = gpd.read_parquet(processed_dir / "hex_grid.parquet").to_crs(METRIC_CRS)

    if region.fault_stub_mode:
        print("[fault] STUB MODE: fault_distance_m set to 0.0 for all hexes.")
        df = pd.DataFrame({"hex_id": hex_gdf["hex_id"], "fault_distance_m": 0.0})
        out_path = processed_dir / "fault_features.parquet"
        df.to_parquet(out_path)
        print(f"[fault] Saved {len(df)} rows (stub mode) -> {out_path}")
        return df

    fault_path = region.fault_csv
    if not fault_path.exists():
        raise FileNotFoundError(f"Fault file not found at {fault_path}.")

    # Branch A: GeoJSON / GPKG vector fault geometries (e.g. Hokkaido gem_active_faults)
    if fault_path.suffix.lower() in (".geojson", ".gpkg", ".shp"):
        print(f"[fault] Loading vector fault features from '{fault_path.name}'...")
        raw_fault_gdf = gpd.read_file(fault_path)

        # Ensure CRS is WGS84 for spatial bounding box filtering
        if raw_fault_gdf.crs is None or not raw_fault_gdf.crs.equals(WGS84):
            raw_fault_gdf = raw_fault_gdf.to_crs(WGS84)

        # Expand region bbox by ~1.0 degree padding margin in WGS84 before reprojecting
        bbox = region.bbox
        pad = 1.0
        padded_bbox_poly = box(
            bbox["min_lon"] - pad,
            bbox["min_lat"] - pad,
            bbox["max_lon"] + pad,
            bbox["max_lat"] + pad,
        )

        # Filter fault traces intersecting the padded bounding box
        # NOTE: This spatial bbox filtering is a temporary isolation mechanism.
        # Ideally, the active fault data should be isolated specifically to the seismogenic fault segment
        # (the Ishikari lowland eastern-margin fault zone near Atsuma/Mukawa/Abira) rather than just bbox-clipped.
        # TODO: Perform manual verification against GSI's official fault model for the 2018 Hokkaido earthquake event.
        filtered_fault_gdf = raw_fault_gdf[raw_fault_gdf.intersects(padded_bbox_poly)]

        if filtered_fault_gdf.empty:
            raise ValueError(
                f"No fault traces in '{fault_path.name}' intersected the padded bounding box for region '{region.name}' "
                f"(bbox: {bbox} with {pad}° padding)."
            )

        print(
            f"[fault] Filtered {len(raw_fault_gdf)} global fault traces down to {len(filtered_fault_gdf)} "
            f"traces within padded bbox for region '{region.name}'."
        )

        fault_gdf = filtered_fault_gdf.to_crs(METRIC_CRS)
        centroids = hex_gdf.geometry.centroid
        distances = centroids.apply(lambda p: fault_gdf.distance(p).min())
        hex_gdf["fault_distance_m"] = distances

    # Branch B: CSV point trace (e.g. Noto point CSV)
    else:
        print(f"[fault] Loading point trace fault features from '{fault_path.name}'...")
        fault_df = pd.read_csv(fault_path)

        # Try common column names
        lon_col = next((c for c in ("lon", "longitude", "X", "FAULT_LON") if c in fault_df.columns), fault_df.columns[0])
        lat_col = next((c for c in ("lat", "latitude", "Y", "FAULT_LAT") if c in fault_df.columns), fault_df.columns[1])

        fault_df = fault_df.sort_values(lon_col)
        fault_line = LineString(zip(fault_df[lon_col], fault_df[lat_col]))
        fault_gdf = gpd.GeoDataFrame(geometry=[fault_line], crs=WGS84).to_crs(METRIC_CRS)
        fault_geom = fault_gdf.geometry.iloc[0]

        hex_gdf["fault_distance_m"] = hex_gdf.geometry.centroid.distance(fault_geom)

    df = hex_gdf[["hex_id", "fault_distance_m"]]
    out_path = processed_dir / "fault_features.parquet"
    df.to_parquet(out_path)
    print(
        f"[fault] Saved {len(df)} rows, distance range "
        f"[{df['fault_distance_m'].min():.0f}, {df['fault_distance_m'].max():.0f}] m -> "
        f"{out_path}"
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build fault features parquet.")
    parser.add_argument("--region", type=str, choices=["noto", "hokkaido"], default="noto", help="Region name (e.g. noto, hokkaido)")
    args = parser.parse_args()

    build_fault_features(get_region(args.region))
