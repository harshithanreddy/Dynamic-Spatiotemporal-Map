"""
Step 5: Ground deformation per hex (local and wide).
Supports both Noto's PIV point product (CSV/GPKG) and Hokkaido's InSAR rasters (qEW/qUD GeoTIFFs).

Output: data/processed/<region_name>/deformation_features.parquet
  columns: hex_id, ground_deformation_local, ground_deformation_wide
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
from config import METRIC_CRS, PROCESSED, RegionConfig, get_region


def load_deformation_points(region: RegionConfig) -> gpd.GeoDataFrame:
    """Loads PIV vector point dataset for regions using point-based deformation (e.g. Noto)."""
    csv_path = region.deformation_csv
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(f"Deformation CSV not found at {csv_path}.")

    df = pd.read_csv(csv_path)
    for col in ("X(m)", "Y(m)"):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in deformation CSV ({csv_path}). "
                f"Available columns: {list(df.columns)}."
            )

    x_min, x_max = df["X(m)"].min(), df["X(m)"].max()
    y_min, y_max = df["Y(m)"].min(), df["Y(m)"].max()
    print(f"[deformation] X(m) range: [{x_min:.2f}, {x_max:.2f}], Y(m) range: [{y_min:.2f}, {y_max:.2f}]")

    if (-180 <= x_min <= 180 and -180 <= x_max <= 180) or (-180 <= y_min <= 180 and -180 <= y_max <= 180):
        print(
            "[deformation] WARNING: X(m) / Y(m) values appear to be in lon/lat degrees range (-180 to 180) "
            "rather than projected metric coordinates!"
        )

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["X(m)"], df["Y(m)"]),
        crs=METRIC_CRS,
    )
    return gdf


def _zonal_mean_tif(tif_path: Path, hex_gdf: gpd.GeoDataFrame) -> pd.Series:
    """Computes zonal mean of a single-band GeoTIFF per hex polygon."""
    with rasterio.open(tif_path) as src:
        raster = src.read(1).astype("float64")
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if nodata is not None:
        raster = np.where(raster == nodata, np.nan, raster)

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
                    val = np.nan
                values.append(val)
    return pd.Series(values)


def build_deformation_features(region: RegionConfig) -> pd.DataFrame:
    """
    Computes ground deformation features per hex.
    Branches based on data format:
      - GeoTIFF rasters (InSAR qEW / qUD format, e.g. Hokkaido)
      - PIV CSV / GPKG vector points (e.g. Noto)
    """
    processed_dir = PROCESSED / region.name
    processed_dir.mkdir(parents=True, exist_ok=True)

    hex_gdf = gpd.read_parquet(processed_dir / "hex_grid.parquet").to_crs(METRIC_CRS)

    # Branch 1: InSAR GeoTIFF Rasters (e.g. Hokkaido)
    if region.deformation_tif_ew is not None and region.deformation_tif_ud is not None:
        print(f"[deformation] Processing InSAR GeoTIFF rasters for region '{region.name}'...")
        if not region.deformation_tif_ew.exists():
            raise FileNotFoundError(f"InSAR EW GeoTIFF not found at {region.deformation_tif_ew}")
        if not region.deformation_tif_ud.exists():
            raise FileNotFoundError(f"InSAR UD GeoTIFF not found at {region.deformation_tif_ud}")

        ew_series = _zonal_mean_tif(region.deformation_tif_ew, hex_gdf)
        ud_series = _zonal_mean_tif(region.deformation_tif_ud, hex_gdf)

        # Note: For Hokkaido (Branch 1), "ground_deformation_local" and "ground_deformation_wide" hold
        # qEW and qUD (east-west and vertical InSAR-derived deformation in cm), respectively. This is a
        # different physical quantity than Noto's PIV-derived local/wide displacement magnitude occupying
        # the same column names — this reuse is intentional for feature-table schema consistency across regions.
        df = pd.DataFrame({
            "hex_id": hex_gdf["hex_id"],
            "ground_deformation_local": ew_series.fillna(0.0),
            "ground_deformation_wide": ud_series.fillna(0.0),
        })

    # Branch 2: PIV Vector Points (e.g. Noto)
    elif region.deformation_csv is not None or region.deformation_gpkg is not None:
        print(f"[deformation] Processing PIV vector point dataset for region '{region.name}'...")
        points = load_deformation_points(region)

        val_col = region.deformation_value_col
        val_sec = region.deformation_value_col_secondary

        cols_to_check = [c for c in (val_col, val_sec) if c is not None]
        for col in cols_to_check:
            if col not in points.columns:
                raise KeyError(
                    f"Column '{col}' not found in deformation CSV. Available columns: {list(points.columns)}."
                )

        select_cols = [c for c in cols_to_check if c in points.columns] + ["geometry"]
        joined = gpd.sjoin(
            points[select_cols],
            hex_gdf[["hex_id", "geometry"]],
            how="inner",
            predicate="within",
        )

        rename_dict = {}
        if val_col:
            rename_dict[val_col] = "ground_deformation_local"
        if val_sec:
            rename_dict[val_sec] = "ground_deformation_wide"

        agg_cols = [c for c in (val_col, val_sec) if c and c in joined.columns]
        agg = joined.groupby("hex_id")[agg_cols].mean().rename(columns=rename_dict)

        df = hex_gdf[["hex_id"]].merge(agg, on="hex_id", how="left")
        for col in ("ground_deformation_local", "ground_deformation_wide"):
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

    else:
        raise ValueError(f"No valid deformation dataset configured for region '{region.name}'.")

    out_path = processed_dir / "deformation_features.parquet"
    df.to_parquet(out_path)
    print(f"[deformation] Saved {len(df)} rows -> {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build deformation features parquet.")
    parser.add_argument("--region", type=str, choices=["noto", "hokkaido"], default="noto", help="Region name (e.g. noto, hokkaido)")
    args = parser.parse_args()

    build_deformation_features(get_region(args.region))
