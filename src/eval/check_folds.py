"""
Diagnostic script to check spatial block CV folds and feature importances for a region.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED, RANDOM_SEED, get_region
from src.eval.spatial_cv import spatial_block_folds


def main():
    parser = argparse.ArgumentParser(description="Check spatial CV folds and feature importances.")
    parser.add_argument("--region", type=str, choices=["noto", "hokkaido"], default="noto", help="Region name (e.g. noto, hokkaido)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of spatial block CV folds (default: 5)")
    args = parser.parse_args()

    region = get_region(args.region)
    processed_dir = PROCESSED / region.name
    features_path = processed_dir / f"{region.name}_features.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Feature table not found at {features_path}. Run build_feature_table.py --region {region.name} first.")

    df = pd.read_parquet(features_path)
    dataset_size = len(df)

    print(f"=== Spatial Block Cross-Validation Folds ({region.name.upper()}, {args.n_folds} Folds) ===")
    folds = list(spatial_block_folds(df, n_folds=args.n_folds, seed=RANDOM_SEED))
    pos_counts = []
    test_sizes = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        test_df = df.iloc[test_idx]
        n_train = len(train_idx)
        n_test = len(test_idx)
        test_pct = n_test / dataset_size if dataset_size > 0 else 0.0
        test_sizes.append(n_test)
        n_pos = int(test_df["label"].sum())
        pos_counts.append(n_pos)
        pos_rate = test_df["label"].mean() if n_test > 0 else 0.0
        print(
            f"Fold {fold_idx}: Train size = {n_train:5d}, Test size = {n_test:5d} ({test_pct:5.1%}), "
            f"Test Positives = {n_pos:3d}, Test Positive Rate = {pos_rate:.2%}"
        )

    low_pos_folds = [i + 1 for i, c in enumerate(pos_counts) if c < 5]
    if low_pos_folds:
        print(
            f"[warn] Folds {low_pos_folds} have fewer than 5 positive test examples "
            f"(too few for reliable metric computation)."
        )

    expected_size = dataset_size / args.n_folds
    size_deviant_folds = [
        i + 1 for i, s in enumerate(test_sizes)
        if s > 2.0 * expected_size or s < (expected_size / 2.0)
    ]
    if size_deviant_folds:
        print(
            f"[warn] Folds {size_deviant_folds} test size deviates by >2x from expected mean "
            f"fold size ({expected_size:.0f} hexes)."
        )

    print(
        f"\n[summary] Positive test count range across folds: "
        f"min = {min(pos_counts)}, max = {max(pos_counts)}"
    )

    print(f"\n=== Random Forest Feature Importances (Full {region.name.upper()} Dataset) ===")
    drop_cols = {"hex_id", "lon", "lat", "label"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["label"]

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    print("Top 10 Features by Importance:")
    for feature, importance in importances.head(10).items():
        print(f"  {feature:<65}: {importance:.4f}")


if __name__ == "__main__":
    main()
