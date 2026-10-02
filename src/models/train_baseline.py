"""
Step 8: Logistic regression + random forest baselines, weighted for class imbalance, evaluated
with spatial block cross-validation.

Output: data/processed/<region_name>/baseline_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED, RANDOM_SEED, get_region
from src.eval.spatial_cv import spatial_block_folds

FEATURE_COLS_EXCLUDE = {"hex_id", "lon", "lat", "label"}


def best_f1_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.nanargmax(f1)
    return thresholds[min(best_idx, len(thresholds) - 1)], f1[best_idx]


def run_model(model, X, y, folds):
    aucs, f1s = [], []
    oof_probs = np.full(len(X), np.nan, dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(folds, 1):
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model.fit(X_train_s, y_train)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        oof_probs[test_idx] = y_prob

        if y_test.nunique() < 2:
            print(f"[warn] Fold {fold_idx} has only 1 class present in test set — skipping AUC metric for fold.")
            continue

        aucs.append(roc_auc_score(y_test, y_prob))
        threshold, f1 = best_f1_threshold(y_test.values, y_prob)
        f1s.append(f1)

    if len(aucs) == 0:
        print("[warn] No valid folds with >= 2 classes in test set. Returning NaN metrics.")
        return {"auc_mean": float("nan"), "auc_std": float("nan"), "f1_mean": float("nan"), "n_folds_used": 0}, oof_probs

    metrics = {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "n_folds_used": len(aucs),
    }
    return metrics, oof_probs


def save_predictions(df, oof_probs, region_name, model_name, processed_dir):
    valid_mask = ~np.isnan(oof_probs)
    if not np.all(valid_mask):
        print(f"[warn] {sum(~valid_mask)} hexes missing OOF prediction for {model_name}")
    
    y_true = df["label"].values
    threshold, _ = best_f1_threshold(y_true[valid_mask], oof_probs[valid_mask])
    y_pred_label = np.zeros(len(df), dtype=int)
    y_pred_label[valid_mask] = (oof_probs[valid_mask] >= threshold).astype(int)

    pred_df = pd.DataFrame({
        "h3_index": df["hex_id"],
        "region": region_name,
        "y_true": y_true,
        "y_pred_prob": oof_probs,
        "y_pred_label": y_pred_label,
    })
    out_path = processed_dir / f"predictions_{model_name}.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"[baseline] Saved predictions -> {out_path} (F1 threshold: {threshold:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline ML models on region feature table.")
    parser.add_argument("--region", type=str, choices=["noto", "hokkaido"], default="noto", help="Region name (e.g. noto, hokkaido)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of spatial block CV folds (default: 5)")
    parser.add_argument("--n-blocks-per-side", type=int, default=None, help="Number of spatial blocks per side (default: region default)")
    args = parser.parse_args()

    region = get_region(args.region)
    processed_dir = PROCESSED / region.name
    processed_dir.mkdir(parents=True, exist_ok=True)

    features_path = processed_dir / f"{region.name}_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Feature table not found at {features_path}. Run build_feature_table.py --region {region.name} first.")

    df = pd.read_parquet(features_path)
    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
    X, y = df[feature_cols], df["label"]

    n_blocks = args.n_blocks_per_side if args.n_blocks_per_side is not None else region.default_n_blocks_per_side
    folds = list(spatial_block_folds(df, n_folds=args.n_folds, n_blocks_per_side=n_blocks, seed=RANDOM_SEED))
    print(
        f"[baseline] {region.name.upper()}: {len(folds)} spatial block folds ({args.n_folds}-fold CV, {n_blocks} blocks/side), "
        f"{len(feature_cols)} features, positive rate {y.mean():.3%}"
    )

    results = {}

    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    metrics_lr, oof_lr = run_model(logreg, X, y, folds)
    results["logistic_regression"] = metrics_lr
    save_predictions(df, oof_lr, region.name, "logistic_regression", processed_dir)
    print(
        f"[baseline] Logistic regression: AUC {results['logistic_regression']['auc_mean']:.3f} "
        f"+/- {results['logistic_regression']['auc_std']:.3f} (folds used: {results['logistic_regression']['n_folds_used']})"
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=None,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    metrics_rf, oof_rf = run_model(rf, X, y, folds)
    results["random_forest"] = metrics_rf
    save_predictions(df, oof_rf, region.name, "random_forest", processed_dir)
    print(
        f"[baseline] Random forest: AUC {results['random_forest']['auc_mean']:.3f} "
        f"+/- {results['random_forest']['auc_std']:.3f} (folds used: {results['random_forest']['n_folds_used']})"
    )

    out_json = processed_dir / "baseline_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[baseline] Saved -> {out_json}")
