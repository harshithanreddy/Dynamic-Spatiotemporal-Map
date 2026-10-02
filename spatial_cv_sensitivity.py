"""
spatial_cv_sensitivity.py

Evaluates spatial block CV sensitivity across block granularities (n_blocks_per_side in [3, 5, 8, 12, 16, 20, 25])
to test for spatial adjacency leakage in landslide susceptibility models (Logistic Regression, Random Forest, Static GCN).

Outputs:
  - outputs/cv_sensitivity_<region>.csv
  - outputs/maps/cv_sensitivity_<region>.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

sys.path.append(str(Path(__file__).resolve().parent))
from config import METRIC_CRS, PROCESSED, RANDOM_SEED, ROOT, WGS84, get_region
from src.eval.spatial_cv import spatial_block_folds
from src.models.train_baseline import FEATURE_COLS_EXCLUDE, best_f1_threshold
from src.models.train_gnn import StaticGCN, build_edge_index

# Select CUDA GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLOCK_SIZES = [3, 5, 8, 12, 16, 20, 25]


def compute_leakage_proxies(df: pd.DataFrame, neighbors: dict, folds: list):
    """
    Computes spatial leakage proxies across spatial CV folds:
      1. mean_test_train_adjacency_fraction: fraction of test hexes with >=1 neighbor in train fold
      2. mean_test_train_distance_m: mean Euclidean distance (meters) from test hex to nearest train hex
      3. zero_pos_folds: count of folds with 0 positive test hexes
    """
    # Project centroids to metric CRS (EPSG:6675) for distance calculation in meters
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=WGS84
    ).to_crs(METRIC_CRS)
    coords_metric = np.column_stack([gdf.geometry.x, gdf.geometry.y])

    adj_fractions = []
    mean_distances_m = []
    zero_pos_folds = 0

    for train_idx, test_idx in folds:
        test_hexes = df["hex_id"].iloc[test_idx].tolist()
        train_hex_set = set(df["hex_id"].iloc[train_idx])
        y_test = df["label"].iloc[test_idx].values

        if (y_test == 1).sum() == 0:
            zero_pos_folds += 1

        # 1. Adjacency fraction
        adj_count = 0
        for h in test_hexes:
            nbrs = neighbors.get(h, [])
            if any(n in train_hex_set for n in nbrs):
                adj_count += 1
        adj_fractions.append(adj_count / len(test_hexes) if len(test_hexes) > 0 else 0.0)

        # 2. Distance in meters using cKDTree
        train_pts = coords_metric[train_idx]
        test_pts = coords_metric[test_idx]
        tree = cKDTree(train_pts)
        dists, _ = tree.query(test_pts, k=1)
        mean_distances_m.append(dists.mean())

    return (
        float(np.mean(adj_fractions)),
        float(np.mean(mean_distances_m)),
        zero_pos_folds,
    )


def train_eval_logistic_regression(df: pd.DataFrame, feature_cols: list, folds: list):
    X, y = df[feature_cols], df["label"]
    oof_probs = np.full(len(df), np.nan, dtype=float)

    for train_idx, test_idx in folds:
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train = y.loc[train_idx]
        if y_train.nunique() < 2:
            print("[warn] Fold skipped: train set has < 2 classes", flush=True)
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
        )
        model.fit(X_train_s, y_train)
        oof_probs[test_idx] = model.predict_proba(X_test_s)[:, 1]

    valid = ~np.isnan(oof_probs)
    if not np.any(valid) or len(np.unique(y.values[valid])) < 2:
        return float("nan"), float("nan")
    y_true = y.values[valid]
    y_prob = oof_probs[valid]

    auc = roc_auc_score(y_true, y_prob)
    thresh, _ = best_f1_threshold(y_true, y_prob)
    f1 = f1_score(y_true, (y_prob >= thresh).astype(int))
    return float(auc), float(f1)


def train_eval_random_forest(df: pd.DataFrame, feature_cols: list, folds: list):
    X, y = df[feature_cols], df["label"]
    oof_probs = np.full(len(df), np.nan, dtype=float)

    for train_idx, test_idx in folds:
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train = y.loc[train_idx]
        if y_train.nunique() < 2:
            print("[warn] Fold skipped: train set has < 2 classes", flush=True)
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=None,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train_s, y_train)
        oof_probs[test_idx] = model.predict_proba(X_test_s)[:, 1]

    valid = ~np.isnan(oof_probs)
    if not np.any(valid) or len(np.unique(y.values[valid])) < 2:
        return float("nan"), float("nan")
    y_true = y.values[valid]
    y_prob = oof_probs[valid]

    auc = roc_auc_score(y_true, y_prob)
    thresh, _ = best_f1_threshold(y_true, y_prob)
    f1 = f1_score(y_true, (y_prob >= thresh).astype(int))
    return float(auc), float(f1)


def train_eval_gcn(df: pd.DataFrame, feature_cols: list, neighbors: dict, folds: list):
    oof_probs = np.full(len(df), np.nan, dtype=float)
    hex_ids = df["hex_id"].tolist()
    edge_index = build_edge_index(hex_ids, neighbors).to(DEVICE)

    X_all = StandardScaler().fit_transform(df[feature_cols].values)
    x = torch.tensor(X_all, dtype=torch.float, device=DEVICE)
    y_tensor = torch.tensor(df["label"].values, dtype=torch.float, device=DEVICE)
    data = Data(x=x, edge_index=edge_index, y=y_tensor)

    for train_idx, test_idx in folds:
        train_mask = torch.zeros(len(df), dtype=torch.bool, device=DEVICE)
        test_mask = torch.zeros(len(df), dtype=torch.bool, device=DEVICE)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        y_train = y_tensor[train_mask]
        pos_count = (y_train == 1).sum().item()
        neg_count = (y_train == 0).sum().item()
        if pos_count == 0 or neg_count == 0:
            print("[warn] Fold skipped: train set has < 2 classes", flush=True)
            continue

        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=DEVICE)

        model = StaticGCN(len(feature_cols)).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(150):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.binary_cross_entropy_with_logits(
                out[train_mask], data.y[train_mask], pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            probs = torch.sigmoid(out[test_mask]).cpu().numpy()

        oof_probs[test_idx] = probs

    valid = ~np.isnan(oof_probs)
    if not np.any(valid) or len(np.unique(df["label"].values[valid])) < 2:
        return float("nan"), float("nan")
    y_true = df["label"].values[valid]
    y_prob = oof_probs[valid]

    auc = roc_auc_score(y_true, y_prob)
    thresh, _ = best_f1_threshold(y_true, y_prob)
    f1 = f1_score(y_true, (y_prob >= thresh).astype(int))
    return float(auc), float(f1)


def run_region_sensitivity(region_name: str, skip_gcn: bool = False, resume: bool = True):
    region = get_region(region_name)
    processed_dir = PROCESSED / region.name
    features_path = processed_dir / f"{region.name}_features.parquet"
    neighbors_path = processed_dir / "hex_neighbors.json"

    if not features_path.exists():
        raise FileNotFoundError(f"Features parquet missing at {features_path}")
    if not neighbors_path.exists():
        raise FileNotFoundError(f"Neighbors map missing at {neighbors_path}")

    df = pd.read_parquet(features_path).reset_index(drop=True)
    with open(neighbors_path) as f:
        neighbors = json.load(f)

    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]

    out_csv = ROOT / "outputs" / f"cv_sensitivity_{region.name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    existing_df = None
    if resume and out_csv.exists():
        try:
            existing_df = pd.read_csv(out_csv)
            print(f"[resume] Loaded {len(existing_df)} existing rows from {out_csv}")
        except Exception:
            existing_df = None

    models_to_run = ["logistic_regression", "random_forest"]
    if not skip_gcn:
        models_to_run.append("gcn")

    results = []

    print(f"\n==================================================")
    print(f"SPATIAL CV SENSITIVITY CHECK: {region.name.upper()}")
    print(f"Device: {DEVICE} | Models: {models_to_run}")
    print(f"==================================================")

    for b in BLOCK_SIZES:
        t0 = time.time()
        folds = list(spatial_block_folds(df, n_folds=5, n_blocks_per_side=b, seed=RANDOM_SEED))
        adj_frac, dist_m, zero_pos = compute_leakage_proxies(df, neighbors, folds)

        for m_name in models_to_run:
            # Check if row already calculated
            if existing_df is not None:
                match = existing_df[
                    (existing_df["region"] == region.name)
                    & (existing_df["model"] == m_name)
                    & (existing_df["n_blocks_per_side"] == b)
                ]
                if not match.empty:
                    row_dict = match.iloc[0].to_dict()
                    results.append(row_dict)
                    print(
                        f"[skip] {region.name.upper()} | {m_name:<20} | blocks: {b:2d} | "
                        f"AUC: {row_dict['auc']:.3f} (already in CSV)",
                        flush=True,
                    )
                    continue

            m_t0 = time.time()
            if m_name == "logistic_regression":
                auc, f1 = train_eval_logistic_regression(df, feature_cols, folds)
            elif m_name == "random_forest":
                auc, f1 = train_eval_random_forest(df, feature_cols, folds)
            elif m_name == "gcn":
                auc, f1 = train_eval_gcn(df, feature_cols, neighbors, folds)

            m_elapsed = time.time() - m_t0

            row = {
                "region": region.name,
                "model": m_name,
                "n_blocks_per_side": b,
                "mean_test_train_adjacency_fraction": float(adj_frac),
                "mean_test_train_distance_m": float(dist_m),
                "auc": float(auc),
                "f1": float(f1),
                "n_folds_with_zero_positives": int(zero_pos),
            }
            results.append(row)

            # Update CSV immediately (resumable)
            res_df = pd.DataFrame(results)
            res_df.to_csv(out_csv, index=False)

            print(
                f"[run]  {region.name.upper()} | {m_name:<20} | blocks: {b:2d} | "
                f"adj_frac: {adj_frac:5.1%} | dist: {dist_m/1000:4.1f}km | "
                f"AUC: {auc:.3f} | F1: {f1:.3f} | zero_pos: {zero_pos} | time: {m_elapsed:.1f}s",
                flush=True,
            )

    final_df = pd.DataFrame(results)
    final_df.to_csv(out_csv, index=False)
    print(f"[saved] -> {out_csv}")
    return final_df


def plot_sensitivity(region_name: str, df: pd.DataFrame):
    """Generates sensitivity plot of AUC vs block size across models."""
    out_map_dir = ROOT / "outputs" / "maps"
    out_map_dir.mkdir(parents=True, exist_ok=True)
    out_plot = out_map_dir / f"cv_sensitivity_{region_name}.png"

    fig, ax1 = plt.subplots(figsize=(9, 6))

    models = df["model"].unique()
    colors = {"logistic_regression": "#2b8cbe", "random_forest": "#e6550d", "gcn": "#31a354"}
    labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gcn": "Static GCN",
    }

    block_sizes = sorted(df["n_blocks_per_side"].unique())

    # Highlight block sizes with fold imbalance (zero positives)
    zero_pos_blocks = (
        df.groupby("n_blocks_per_side")["n_folds_with_zero_positives"].max()
    )
    for b in block_sizes:
        if zero_pos_blocks.get(b, 0) > 0:
            ax1.axvspan(b - 0.6, b + 0.6, color="#fee0d2", alpha=0.5, zorder=0)

    for m in models:
        m_df = df[df["model"] == m].sort_values("n_blocks_per_side")
        ax1.plot(
            m_df["n_blocks_per_side"],
            m_df["auc"],
            marker="o",
            linewidth=2,
            color=colors.get(m, "gray"),
            label=labels.get(m, m),
        )

    ax1.set_xlabel("Spatial Blocks per Side (n_blocks_per_side)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Out-of-Fold AUC-ROC", fontsize=11, fontweight="bold")
    ax1.set_xticks(block_sizes)
    ax1.set_ylim(0.70, 1.00)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Secondary twin axis showing test-train adjacency fraction
    ax2 = ax1.twinx()
    adj_df = df.groupby("n_blocks_per_side")["mean_test_train_adjacency_fraction"].mean().reset_index()
    ax2.plot(
        adj_df["n_blocks_per_side"],
        adj_df["mean_test_train_adjacency_fraction"] * 100,
        color="#7570b3",
        linestyle=":",
        linewidth=1.8,
        label="Test-Train Adjacency %",
    )
    ax2.set_ylabel("Test-Train Adjacency % (Proxy)", color="#7570b3", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 105)

    # Title & Legends
    plt.title(
        f"Spatial CV Block-Size Sensitivity & Adjacency Leakage Check ({region_name.upper()})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved sensitivity plot -> {out_plot}")


def print_verdicts_and_summary(all_dfs: list):
    """
    Evaluates fixed 0.02 threshold verdict per model per region and prints summary table.
    """
    combined = pd.concat(all_dfs, ignore_index=True)

    print("\n" + "=" * 80)
    print("SPATIAL BLOCK-SIZE SENSITIVITY & ADJACENCY LEAKAGE VERDICT SUMMARY")
    print("Fixed Threshold Criterion: AUC Delta (Finest block=25 vs Coarsest block=3) > 0.02")
    print("=" * 80)

    verdicts = []

    for region_name in combined["region"].unique():
        reg_df = combined[combined["region"] == region_name]
        print(f"\n--- Region: {region_name.upper()} ---")

        for m_name in reg_df["model"].unique():
            m_df = reg_df[reg_df["model"] == m_name].sort_values("n_blocks_per_side")
            m_df_valid = m_df.dropna(subset=["auc"])

            if m_df_valid.empty:
                print(f"Model: {m_name:<20} - No valid folds across all block sizes")
                continue

            b_max = m_df_valid["n_blocks_per_side"].max()
            b_min = m_df_valid["n_blocks_per_side"].min()

            auc_finest = m_df_valid[m_df_valid["n_blocks_per_side"] == b_max]["auc"].values[0]
            auc_coarsest = m_df_valid[m_df_valid["n_blocks_per_side"] == b_min]["auc"].values[0]
            auc_delta = auc_finest - auc_coarsest

            is_sensitive = auc_delta > 0.02

            if is_sensitive:
                status_str = "FLAGGED: Block-size Sensitive (Potential Leakage)"
                verdict_msg = (
                    f"AUC is block-size sensitive — original CV result likely optimistic due to adjacency leakage."
                )
            else:
                status_str = "PASSED: Stable across Block Sizes"
                verdict_msg = (
                    f"AUC is stable across block sizes — leakage explanation is not supported by this test."
                )

            verdicts.append(
                {
                    "region": region_name.upper(),
                    "model": m_name,
                    "auc_finest_b25": auc_finest,
                    "auc_coarsest_b3": auc_coarsest,
                    "auc_delta": auc_delta,
                    "status": status_str,
                    "verdict": verdict_msg,
                }
            )

            print(f"Model: {m_name:<20}")
            print(f"  Finest Granularity  (b={b_max:2d}) AUC : {auc_finest:.3f}")
            print(f"  Coarsest Granularity(b={b_min:2d}) AUC : {auc_coarsest:.3f}")
            print(f"  AUC Delta (Finest - Coarsest)     : {auc_delta:+.3f}")
            print(f"  Status                             : {status_str}")
            print(f"  Verdict                            : {verdict_msg}\n")

    print("=" * 80)
    print("PIPELINE SUMMARY TABLE (AUC Across Block Granularities)")
    print("=" * 80)

    pivot = combined.pivot_table(
        index=["region", "model"],
        columns="n_blocks_per_side",
        values="auc",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run spatial CV block-size sensitivity check for Noto and Hokkaido."
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["noto", "hokkaido", "both"],
        default="both",
        help="Target region ('noto', 'hokkaido', or 'both')",
    )
    parser.add_argument(
        "--skip-gcn",
        action="store_true",
        help="Skip Static GCN model to speed up baseline execution",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resuming from existing CSV output",
    )
    args = parser.parse_args()

    regions = ["noto", "hokkaido"] if args.region == "both" else [args.region]
    resume = not args.no_resume

    dfs = []
    for r in regions:
        df_res = run_region_sensitivity(r, skip_gcn=args.skip_gcn, resume=resume)
        plot_sensitivity(r, df_res)
        dfs.append(df_res)

    print_verdicts_and_summary(dfs)


if __name__ == "__main__":
    main()
