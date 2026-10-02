"""
Step 3: Assign each hex its dominant (largest-overlap-area) lithology class from the GSJ
seamless geology map.

Output: data/processed/<region_name>/geology_features.parquet
  columns: hex_id, geology_class (categorical, kept as string -> one-hot encode later in
  build_feature_table.py)
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import (
    GEOLOGY_CLASS_COL,
    GEOLOGY_LEGEND,
    GEOLOGY_LITHOLOGY_LEVEL,
    GEOLOGY_POLY,
    METRIC_CRS,
    PROCESSED,
    RegionConfig,
    get_region,
)


def build_geology_features(region: RegionConfig) -> pd.DataFrame:
    """
    Assigns each hex cell its dominant geology class from the seamless geology dataset.
    Note: Geology shapefile and legend are national/global datasets.
    """
    processed_dir = PROCESSED / region.name
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not GEOLOGY_POLY.exists():
        raise FileNotFoundError(f"Geology shapefile not found at {GEOLOGY_POLY}.")
    if not GEOLOGY_LEGEND.exists():
        raise FileNotFoundError(f"Geology legend file not found at {GEOLOGY_LEGEND}.")

    hex_gdf = gpd.read_parquet(processed_dir / "hex_grid.parquet").to_crs(METRIC_CRS)
    geo_gdf = gpd.read_file(GEOLOGY_POLY).to_crs(METRIC_CRS)

    if GEOLOGY_CLASS_COL not in geo_gdf.columns:
        raise KeyError(
            f"Column '{GEOLOGY_CLASS_COL}' not found in geology shapefile. "
            f"Available columns: {list(geo_gdf.columns)}. "
            f"Update GEOLOGY_CLASS_COL in config.py to the correct lithology column name."
        )

    legend_df = pd.read_csv(GEOLOGY_LEGEND, sep="\t")
    if GEOLOGY_CLASS_COL not in legend_df.columns:
        raise KeyError(
            f"Column '{GEOLOGY_CLASS_COL}' not found in legend file {GEOLOGY_LEGEND}. "
            f"Available columns: {list(legend_df.columns)}."
        )
    if GEOLOGY_LITHOLOGY_LEVEL not in legend_df.columns:
        raise KeyError(
            f"Column '{GEOLOGY_LITHOLOGY_LEVEL}' not found in legend file {GEOLOGY_LEGEND}. "
            f"Available columns: {list(legend_df.columns)}."
        )

    geo_gdf = geo_gdf.merge(
        legend_df[[GEOLOGY_CLASS_COL, GEOLOGY_LITHOLOGY_LEVEL]],
        on=GEOLOGY_CLASS_COL,
        how="left",
    )

    n_unmatched = geo_gdf[GEOLOGY_LITHOLOGY_LEVEL].isna().sum()
    if n_unmatched > 0:
        print(f"[geology] {n_unmatched} shapefile polygons had no legend match -> filled 'unknown'")
        geo_gdf[GEOLOGY_LITHOLOGY_LEVEL] = geo_gdf[GEOLOGY_LITHOLOGY_LEVEL].fillna("unknown")

    # Overlay hex grid with geology polygons, compute intersection area, keep the class with the
    # largest overlap area per hex.
    hex_gdf = hex_gdf[["hex_id", "geometry"]]
    geo_gdf = geo_gdf[[GEOLOGY_LITHOLOGY_LEVEL, "geometry"]]

    overlay = gpd.overlay(hex_gdf, geo_gdf, how="intersection")
    overlay["area"] = overlay.geometry.area

    dominant = (
        overlay.sort_values("area", ascending=False)
        .drop_duplicates(subset="hex_id", keep="first")[["hex_id", GEOLOGY_LITHOLOGY_LEVEL]]
        .rename(columns={GEOLOGY_LITHOLOGY_LEVEL: "geology_class"})
    )

    # Hexes with no geology overlap (e.g. small slivers at the coast) -> fill with "unknown"
    df = hex_gdf[["hex_id"]].merge(dominant, on="hex_id", how="left")
    n_missing = df["geology_class"].isna().sum()
    if n_missing:
        print(f"[geology] {n_missing} hexes with no geology overlap -> filled 'unknown'")
        df["geology_class"] = df["geology_class"].fillna("unknown")

    out_path = processed_dir / "geology_features.parquet"
    df.to_parquet(out_path)
    print(
        f"[geology] Saved {len(df)} rows, {df['geology_class'].nunique()} classes -> "
        f"{out_path}"
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build geology features parquet.")
    parser.add_argument("--region", type=str, choices=["noto", "hokkaido"], default="noto", help="Region name (e.g. noto, hokkaido)")
    args = parser.parse_args()

    build_geology_features(get_region(args.region))
