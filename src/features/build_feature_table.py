"""
Step 7: Merge terrain, geology, fault, deformation features and labels into one table, ready for
modeling. One-hot encodes geology_class.

Output: data/processed/<region_name>/<region_name>_features.parquet
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED, RegionConfig, get_region
from src.features.deformation_features import build_deformation_features
from src.features.fault_features import build_fault_features
from src.features.geology_features import build_geology_features
from src.features.labels import build_labels
from src.features.terrain_features import build_terrain_features


def build_feature_table(region: RegionConfig) -> pd.DataFrame:
    """
    Builds and merges all feature parquets and labels for the given region into a consolidated feature table.
    Drops hexes missing label data (unlabeled / outside inventory coverage) prior to feature imputation.
    """
    processed_dir = PROCESSED / region.name
    processed_dir.mkdir(parents=True, exist_ok=True)

    hex_grid_path = processed_dir / "hex_grid.parquet"
    if not hex_grid_path.exists():
        raise FileNotFoundError(
            f"Hex grid not found at {hex_grid_path}. "
            f"Run build_h3_grid.py --region {region.name} first."
        )

    # Ensure individual feature tables exist, build lazily if missing
    terrain_path = processed_dir / "terrain_features.parquet"
    if not terrain_path.exists():
        build_terrain_features(region)

    geology_path = processed_dir / "geology_features.parquet"
    if not geology_path.exists():
        build_geology_features(region)

    fault_path = processed_dir / "fault_features.parquet"
    if not fault_path.exists():
        build_fault_features(region)

    deformation_path = processed_dir / "deformation_features.parquet"
    if not deformation_path.exists():
        build_deformation_features(region)

    labels_path = processed_dir / "labels.parquet"
    if not labels_path.exists():
        build_labels(region)

    hex_grid = pd.read_parquet(hex_grid_path)[["hex_id", "lon", "lat"]]
    terrain = pd.read_parquet(terrain_path)
    geology = pd.read_parquet(geology_path)
    fault = pd.read_parquet(fault_path)
    deformation = pd.read_parquet(deformation_path)
    labels = pd.read_parquet(labels_path)[["hex_id", "label"]]

    df = hex_grid
    for other in (terrain, geology, fault, deformation, labels):
        df = df.merge(other, on="hex_id", how="left")

    n_before = len(df)
    assert len(df) == len(hex_grid), "Row count changed during merge — check for duplicate hex_ids"

    # Drop hexes with missing label values (no inventory coverage)
    n_unlabeled = df["label"].isna().sum()
    if n_unlabeled > 0:
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        print(
            f"[merge] Dropping {n_unlabeled} hexes with no landslide-inventory label "
            f"coverage ({n_unlabeled / n_before:.1%} of grid)."
        )

    df["label"] = df["label"].astype(int)

    # One-hot encode geology class, drop rare classes (<1% of hexes) into an "other" bucket
    class_counts = df["geology_class"].value_counts(normalize=True)
    rare_classes = class_counts[class_counts < 0.01].index
    df["geology_class"] = df["geology_class"].replace({c: "other" for c in rare_classes})
    df = pd.get_dummies(df, columns=["geology_class"], prefix="geo")

    # Basic missing-value handling for feature columns
    missing_report = df.isna().sum()
    missing_report = missing_report[missing_report > 0]
    if len(missing_report):
        print(f"[merge] Missing values before imputation:\n{missing_report}")
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "label"]
        for col in numeric_cols:
            med = df[col].median()
            fill_val = 0.0 if pd.isna(med) else med
            df[col] = df[col].fillna(fill_val)

    out_path = processed_dir / f"{region.name}_features.parquet"
    df.to_parquet(out_path)
    print(f"[merge] Saved {len(df)} rows, {len(df.columns)} columns -> {out_path}")
    print(f"[merge] Positive label rate: {df['label'].mean():.3%}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build consolidated feature table parquet.")
    parser.add_argument("--region", type=str, choices=["noto", "hokkaido"], default="noto", help="Region name (e.g. noto, hokkaido)")
    args = parser.parse_args()

    build_feature_table(get_region(args.region))
