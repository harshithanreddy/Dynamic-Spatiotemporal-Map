"""
Step 2: Compute elevation, slope, aspect, curvature from the DEM and aggregate (mean) per hex.

Output: data/processed/<region_name>/terrain_features.parquet
  columns: hex_id, elevation, slope, aspect, curvature
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED, RegionConfig, get_region


def compute_terrain_rasters(dem_path: Path):
    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype("float64")
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        bounds = src.bounds

    if nodata is not None:
        elevation = np.where(elevation == nodata, np.nan, elevation)

    dx = transform[0]
    dy = abs(transform[4])

    if crs and crs.is_geographic:
        center_lat = (bounds.bottom + bounds.top) / 2.0
        dx = dx * 111139.0 * np.cos(np.radians(center_lat))
        dy = dy * 111139.0

    dzdy, dzdx = np.gradient(elevation, dy, dx)

    slope_degrees = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    aspect_degrees = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = (450 - aspect_degrees) % 360
    aspect = np.where(np.isclose(slope_degrees, 0, atol=1e-5), np.nan, aspect)

    _, d2z_dx2 = np.gradient(dzdx, dy, dx)
    d2z_dy2, _ = np.gradient(dzdy, dy, dx)
    curvature = d2z_dx2 + d2z_dy2

    return {
        "elevation": elevation,
        "slope": slope_degrees,
        "aspect": aspect,
        "curvature": curvature,
    }, transform, crs


def zonal_mean(raster: np.ndarray, transform, crs, hex_gdf: gpd.GeoDataFrame, band_name: str) -> pd.Series:
    """Mean raster value within each hex polygon, via rasterio.mask on a small in-memory dataset."""
    from rasterio.io import MemoryFile

    values = []
    profile = {
        "driver": "GTiff",
        "height": raster.shape[0],
        "width": raster.shape[1],
        "count": 1,
        "dtype": raster.dtype,
        "crs": crs,
        "transform": transform,
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(raster, 1)
            hex_gdf_proj = hex_gdf.to_crs(crs)
            for geom in hex_gdf_proj.geometry:
                try:
                    out, _ = mask(dataset, [geom], crop=True, nodata=np.nan)
                    val = np.nanmean(out)
                except ValueError:
                    val = np.nan  # hex falls entirely outside raster extent
                values.append(val)
    return pd.Series(values, name=band_name)


def zonal_circular_mean(raster_degrees: np.ndarray, transform, crs, hex_gdf: gpd.GeoDataFrame, band_name: str) -> pd.Series:
    sin_raster = np.sin(np.radians(raster_degrees))
    cos_raster = np.cos(np.radians(raster_degrees))
    sin_mean = zonal_mean(sin_raster, transform, crs, hex_gdf, "sin_temp")
    cos_mean = zonal_mean(cos_raster, transform, crs, hex_gdf, "cos_temp")
    angle = np.degrees(np.arctan2(sin_mean.values, cos_mean.values)) % 360
    return pd.Series(angle, name=band_name)


def build_terrain_features(region: RegionConfig) -> pd.DataFrame:
    """
    Computes terrain features (elevation, slope, aspect, curvature) for hex grid from region DEM.
    """
    processed_dir = PROCESSED / region.name
    processed_dir.mkdir(parents=True, exist_ok=True)

    dem_path = region.dem

    if not dem_path.exists():
        raise FileNotFoundError(
            f"DEM not found at {dem_path}. Place your DEM there or update config. "
            f"See README section 1 for expected layout."
        )

    with rasterio.open(dem_path) as src:
        print(f"[terrain] DEM bounds (left, bottom, right, top): {src.bounds}")
        print(f"[terrain] DEM CRS: {src.crs}")

    hex_gdf = gpd.read_parquet(processed_dir / "hex_grid.parquet")
    rasters, transform, crs = compute_terrain_rasters(dem_path)

    df = pd.DataFrame({"hex_id": hex_gdf["hex_id"]})
    for name, raster in rasters.items():
        if name == "aspect":
            df[name] = zonal_circular_mean(raster, transform, crs, hex_gdf, name).values
        else:
            df[name] = zonal_mean(raster, transform, crs, hex_gdf, name).values
        n_missing = df[name].isna().sum()
        if n_missing:
            print(f"[terrain] {n_missing} hexes with no {name} value (outside DEM extent) -> will need imputation")

    out_path = processed_dir / "terrain_features.parquet"
    df.to_parquet(out_path)
    print(f"[terrain] Saved {len(df)} rows -> {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build terrain features parquet.")
    parser.add_argument("--region", type=str, choices=["noto", "hokkaido"], default="noto", help="Region name (e.g. noto, hokkaido)")
    args = parser.parse_args()

    build_terrain_features(get_region(args.region))
