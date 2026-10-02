"""
Step 6: Binary landslide label per hex — 1 if landslide coverage fraction >= LABEL_COVERAGE_THRESHOLD, else 0.

Output: data/processed/<region_name>/labels.parquet
  columns: hex_id, label, landslide_area_m2, hex_area_m2, coverage_fraction
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import LABEL_COVERAGE_THRESHOLD, METRIC_CRS, PROCESSED, RegionConfig, get_region


def build_labels(region: RegionConfig) -> pd.DataFrame:
    """
    Computes binary landslide label per hex grid cell based on landslide polygon coverage.
    """
    processed_dir = PROCESSED / region.name
    processed_dir.mkdir(parents=True, exist_ok=True)

    landslide_paths = (
        region.landslide_path
        if isinstance(region.landslide_path, list)
        else [region.landslide_path]
    )

    gdfs = []
    for path in landslide_paths:
        if not path.exists():
            raise FileNotFoundError(f"Landslide polygon file not found at {path}.")
        try:
            gdf = gpd.read_file(path, layer="landslidepolygon").to_crs(METRIC_CRS)
        except Exception:
            gdf = gpd.read_file(path).to_crs(METRIC_CRS)
        gdfs.append(gdf)

    landslide_gdf = pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
    
    # Ensure only Polygon/MultiPolygon geometries are passed to geopandas.overlay
    landslide_gdf = landslide_gdf[landslide_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    landslide_gdf = landslide_gdf.explode(index_parts=False).reset_index(drop=True)

    hex_gdf = gpd.read_parquet(processed_dir / "hex_grid.parquet").to_crs(METRIC_CRS)

    overlay = gpd.overlay(
        hex_gdf[["hex_id", "geometry"]],
        landslide_gdf[["geometry"]],
        how="intersection",
    )
    overlay["landslide_area_m2"] = overlay.geometry.area
    area_agg = overlay.groupby("hex_id")["landslide_area_m2"].sum().reset_index()

    hex_gdf = hex_gdf.merge(area_agg, on="hex_id", how="left")
    hex_gdf["landslide_area_m2"] = hex_gdf["landslide_area_m2"].fillna(0)
    hex_gdf["hex_area_m2"] = hex_gdf.geometry.area
    hex_gdf["coverage_fraction"] = hex_gdf["landslide_area_m2"] / hex_gdf["hex_area_m2"]
    hex_gdf["label"] = (hex_gdf["coverage_fraction"] >= LABEL_COVERAGE_THRESHOLD).astype(int)

    df = hex_gdf[["hex_id", "label", "landslide_area_m2", "hex_area_m2", "coverage_fraction"]]
    pos_rate = df["label"].mean()
    out_path = processed_dir / "labels.parquet"
    df.to_parquet(out_path)

    print("[labels] Coverage fraction distribution:")
    print(hex_gdf["coverage_fraction"].describe())
    print(
        f"[labels] Saved {len(df)} rows, positive rate {pos_rate:.3%} "
        f"({df['label'].sum()} positive hexes at threshold >= {LABEL_COVERAGE_THRESHOLD:.0%}) -> "
        f"{out_path}"
    )

    if pos_rate > 0.35 or pos_rate < 0.001:
        print(
            f"[warn] Positive rate {pos_rate:.3%} looks off — sanity check the landslide "
            f"polygon file and H3 resolution before moving on."
        )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build landslide label parquet.")
    parser.add_argument("--region", type=str, choices=["noto", "hokkaido"], default="noto", help="Region name (e.g. noto, hokkaido)")
    args = parser.parse_args()

    build_labels(get_region(args.region))
