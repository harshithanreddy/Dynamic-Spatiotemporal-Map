"""
visualize_predictions.py

Visualizes ground truth vs predicted landslide susceptibility on the H3 hex grid for Noto and Hokkaido regions.

Usage:
    python visualize_predictions.py --region noto --model random_forest
    python visualize_predictions.py --region hokkaido --model logistic_regression
    python visualize_predictions.py --region both --model gcn --dpi 300
"""
argparse_description = "Visualize ground truth vs predicted landslide susceptibility maps."

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED, ROOT, WGS84, get_region


MODEL_NAME_MAP = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gcn": "Static GCN",
}


def ensure_predictions_exist(region_name: str, model_name: str) -> Path:
    """
    Strictly validates existence of predictions_<model>.csv; fails loudly if missing
    without retraining.
    """
    processed_dir = PROCESSED / region_name
    pred_path = processed_dir / f"predictions_{model_name}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"ERROR: Predictions file not found at {pred_path}.\n"
            f"Retraining is strictly disabled in visualize_predictions.py.\n"
            f"Please run the appropriate training script (train_baseline.py or train_gnn.py) "
            f"for region '{region_name}' and model '{model_name}' first."
        )
    return pred_path


def verify_against_baseline_results(region_name: str, model_name: str, computed_auc: float, tolerance: float = 1e-6):
    """
    Asserts that computed OOF AUC matches baseline_results_v2.csv within tolerance.
    Raises AssertionError if metrics diverge beyond tolerance.
    """
    results_v2_path = ROOT / "outputs" / "baseline_results_v2.csv"
    if not results_v2_path.exists():
        print(f"[vis warn] baseline_results_v2.csv not found at {results_v2_path}; skipping divergence check.")
        return

    df_v2 = pd.read_csv(results_v2_path)
    match = df_v2[(df_v2["region"] == region_name) & (df_v2["model"] == model_name)]
    if match.empty:
        print(f"[vis warn] No entry for region='{region_name}', model='{model_name}' in baseline_results_v2.csv; skipping check.")
        return

    expected_auc = match.iloc[0]["auc_mean"]
    diff = abs(computed_auc - expected_auc)
    if diff > tolerance:
        raise AssertionError(
            f"REPRODUCIBILITY BUG DETECTED: Computed AUC ({computed_auc:.6f}) for region '{region_name}', "
            f"model '{model_name}' diverges from baseline_results_v2.csv expected AUC ({expected_auc:.6f}) "
            f"by {diff:.8f} (tolerance: {tolerance:.1e})!"
        )
    print(f"[vis verify] Assertion passed: computed AUC ({computed_auc:.6f}) matches baseline_results_v2.csv ({expected_auc:.6f}) within {tolerance:.1e}.")


def load_region_gdf(region_name: str, model_name: str) -> gpd.GeoDataFrame:
    """
    Loads grid geometry and model predictions for a region, validates schema integrity,
    and returns a merged GeoDataFrame reprojected to EPSG:4326 (WGS84).
    """
    processed_dir = PROCESSED / region_name
    grid_path = processed_dir / "hex_grid.parquet"
    if not grid_path.exists():
        raise FileNotFoundError(
            f"Hex grid missing at {grid_path}. Run build_h3_grid.py --region {region_name} first."
        )

    pred_path = ensure_predictions_exist(region_name, model_name)

    grid_gdf = gpd.read_parquet(grid_path)
    pred_df = pd.read_csv(pred_path)

    # Validate predictions schema & completeness
    required_cols = {"h3_index", "region", "y_true", "y_pred_prob", "y_pred_label"}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Predictions file {pred_path} missing required columns: {missing}")

    # Assertion 1: No duplicate h3_index in predictions
    if pred_df["h3_index"].duplicated().any():
        n_dups = pred_df["h3_index"].duplicated().sum()
        raise AssertionError(f"Predictions file {pred_path} contains {n_dups} duplicate h3_index entries.")

    # Assertion 2: All h3_index in predictions exist in hex_grid
    grid_hexes = set(grid_gdf["hex_id"])
    pred_hexes = set(pred_df["h3_index"])
    unknown_hexes = pred_hexes - grid_hexes
    if unknown_hexes:
        raise AssertionError(
            f"Found {len(unknown_hexes)} h3_indexes in {pred_path} that do not exist in {grid_path}!"
        )

    # Assertion 3: No NaN values in prediction columns for labeled hexes
    for col in ["y_true", "y_pred_prob", "y_pred_label"]:
        if pred_df[col].isna().any():
            raise AssertionError(f"Predictions file {pred_path} contains NaN in column '{col}'!")

    # Left-join grid with predictions to retain excluded/unlabeled hexes without dropping rows
    merged = grid_gdf.merge(pred_df, left_on="hex_id", right_on="h3_index", how="left")

    # Flag unlabeled/excluded hexes (~19% of Hokkaido grid)
    merged["is_unlabeled"] = merged["y_pred_prob"].isna()

    # Ensure output is reprojected to WGS84 (EPSG:4326)
    merged = merged.to_crs(WGS84)

    return merged


def compute_metrics_and_confusion(gdf: gpd.GeoDataFrame):
    """Computes AUC, F1, positive rate, and confusion matrix counts for labeled hexes."""
    labeled = gdf[~gdf["is_unlabeled"]].copy()
    y_true = labeled["y_true"].astype(int).values
    y_prob = labeled["y_pred_prob"].values
    y_pred = labeled["y_pred_label"].astype(int).values

    total_hexes = len(gdf)
    n_labeled = len(labeled)
    n_unlabeled = gdf["is_unlabeled"].sum()

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    f1 = f1_score(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    pos_rate = y_true.mean()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / n_labeled if n_labeled > 0 else 0.0

    return {
        "total_hexes": total_hexes,
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "unlabeled_pct": n_unlabeled / total_hexes * 100.0,
        "pos_rate": pos_rate,
        "auc": auc,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
    }


def print_summary(region_name: str, model_name: str, metrics: dict):
    """Prints clean, formatted summary of metrics & confusion counts to stdout."""
    model_title = MODEL_NAME_MAP.get(model_name, model_name)
    print("=" * 60)
    print(f"PREDICTION VISUALIZATION SUMMARY")
    print(f"Region: {region_name.upper()} | Model: {model_title}")
    print("-" * 60)
    print(f"Total Grid Hexes   : {metrics['total_hexes']:,}")
    print(f"Labeled Hexes      : {metrics['n_labeled']:,}")
    print(f"Unlabeled/Excluded : {metrics['n_unlabeled']:,} ({metrics['unlabeled_pct']:.1f}%)")
    print(f"Positive Label Rate: {metrics['pos_rate']:.3%} ({metrics['tp'] + metrics['fn']:,} positive hexes)")
    print()
    print("Performance Metrics (Out-of-Fold Spatial CV):")
    print(f"  AUC-ROC          : {metrics['auc']:.3f}")
    print(f"  F1 Score         : {metrics['f1']:.3f}")
    print(f"  Overall Accuracy : {metrics['accuracy']:.1%}")
    print()
    print("Confusion Matrix Breakdown (Hex Counts):")
    print(f"  True Positives  (TP) : {metrics['tp']:,}")
    print(f"  False Positives (FP) : {metrics['fp']:,}")
    print(f"  False Negatives (FN) : {metrics['fn']:,}")
    print(f"  True Negatives  (TN) : {metrics['tn']:,}")
    print("=" * 60)


def plot_row(axes, gdf: gpd.GeoDataFrame, region_name: str, model_name: str, metrics: dict):
    """Plots a 4-panel row for a single region (Ground Truth, Pred Prob, Pred Binary, Metrics)."""
    model_title = MODEL_NAME_MAP.get(model_name, model_name)
    reg_title = region_name.capitalize()

    color_neg = "#4575b4"
    color_pos = "#d73027"
    color_nodata = "#808080"

    labeled_gdf = gdf[~gdf["is_unlabeled"]].copy()
    unlabeled_gdf = gdf[gdf["is_unlabeled"]].copy()

    # Map 1: Ground Truth
    ax_gt = axes[0]
    ax_gt.set_title(f"{reg_title} - Ground Truth\n(Positive Rate: {metrics['pos_rate']:.1%})", fontsize=11, fontweight="bold")
    if not unlabeled_gdf.empty:
        unlabeled_gdf.plot(ax=ax_gt, color=color_nodata, edgecolor="none", linewidth=0)
    labeled_gdf.plot(
        ax=ax_gt,
        column="y_true",
        categorical=True,
        cmap=mcolors.ListedColormap([color_neg, color_pos]),
        edgecolor="none",
        linewidth=0,
    )
    ax_gt.set_axis_off()

    legend_handles = [
        mpatches.Patch(color=color_neg, label="Negative (0)"),
        mpatches.Patch(color=color_pos, label="Positive (1)"),
    ]
    if not unlabeled_gdf.empty:
        legend_handles.append(mpatches.Patch(color=color_nodata, label="No Data / Excluded"))
    ax_gt.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.9)

    # Map 2: Predicted Probability
    ax_prob = axes[1]
    ax_prob.set_title(f"{reg_title} - Pred Probability\n({model_title} | AUC: {metrics['auc']:.3f})", fontsize=11, fontweight="bold")
    if not unlabeled_gdf.empty:
        unlabeled_gdf.plot(ax=ax_prob, color=color_nodata, edgecolor="none", linewidth=0)
    labeled_gdf.plot(
        ax=ax_prob,
        column="y_pred_prob",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolor="none",
        linewidth=0,
    )
    ax_prob.set_axis_off()

    # Map 3: Predicted Binary Label
    ax_bin = axes[2]
    ax_bin.set_title(f"{reg_title} - Pred Binary Label\n(F1 Score: {metrics['f1']:.3f})", fontsize=11, fontweight="bold")
    if not unlabeled_gdf.empty:
        unlabeled_gdf.plot(ax=ax_bin, color=color_nodata, edgecolor="none", linewidth=0)
    labeled_gdf.plot(
        ax=ax_bin,
        column="y_pred_label",
        categorical=True,
        cmap=mcolors.ListedColormap([color_neg, color_pos]),
        edgecolor="none",
        linewidth=0,
    )
    ax_bin.set_axis_off()

    # Panel 4: Metrics & Confusion Matrix
    ax_cm = axes[3]
    ax_cm.axis("off")
    ax_cm.set_title(f"{reg_title} Metrics & Confusion", fontsize=11, fontweight="bold")

    info_text = (
        f"Region: {reg_title}\n"
        f"Model: {model_title}\n"
        f"---------------------------\n"
        f"AUC-ROC   : {metrics['auc']:.3f}\n"
        f"F1 Score  : {metrics['f1']:.3f}\n"
        f"Accuracy  : {metrics['accuracy']:.1%}\n"
        f"---------------------------\n"
        f"Confusion Matrix Counts:\n"
        f"  TP (True Pos)  : {metrics['tp']:,}\n"
        f"  FP (False Pos) : {metrics['fp']:,}\n"
        f"  FN (False Neg) : {metrics['fn']:,}\n"
        f"  TN (True Neg)  : {metrics['tn']:,}\n"
        f"---------------------------\n"
        f"Total Hexes : {metrics['total_hexes']:,}\n"
        f"  Labeled   : {metrics['n_labeled']:,}\n"
        f"  Excluded  : {metrics['n_unlabeled']:,} ({metrics['unlabeled_pct']:.1f}%)"
    )

    ax_cm.text(
        0.05,
        0.5,
        info_text,
        transform=ax_cm.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9),
    )


def generate_visualization(region_name: str, model_name: str, dpi: int = 300) -> Path:
    """Main visualization orchestrator for 'noto', 'hokkaido', or 'both'."""
    output_dir = ROOT / "outputs" / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    regions_to_plot = ["noto", "hokkaido"] if region_name == "both" else [region_name]
    gdfs = {}
    metrics_map = {}

    for r in regions_to_plot:
        gdf = load_region_gdf(r, model_name)
        metrics = compute_metrics_and_confusion(gdf)
        verify_against_baseline_results(r, model_name, metrics["auc"], tolerance=1e-6)
        gdfs[r] = gdf
        metrics_map[r] = metrics
        print_summary(r, model_name, metrics)

    n_rows = len(regions_to_plot)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(22, 5.5 * n_rows),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.7]},
        squeeze=False,
    )

    model_title = MODEL_NAME_MAP.get(model_name, model_name)
    fig.suptitle(
        f"Landslide Susceptibility Comparison: Ground Truth vs {model_title} Predictions",
        fontsize=14,
        fontweight="bold",
        y=0.98 if n_rows == 1 else 0.99,
    )

    for row_idx, r in enumerate(regions_to_plot):
        plot_row(axes[row_idx], gdfs[r], r, model_name, metrics_map[r])

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    cbar_ax = fig.add_axes([0.35, 0.02, 0.30, 0.02])
    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Predicted Landslide Probability", fontsize=10, fontweight="bold")

    out_file = output_dir / f"{region_name}_{model_name}_comparison.png"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[vis] Saved high-resolution map figure -> {out_file} (DPI {dpi})")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=argparse_description)
    parser.add_argument(
        "--region",
        type=str,
        choices=["noto", "hokkaido", "both"],
        default="noto",
        help="Target region ('noto', 'hokkaido', or 'both')",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logistic_regression", "random_forest", "gcn"],
        default="random_forest",
        help="Target model ('logistic_regression', 'random_forest', or 'gcn')",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output PNG resolution DPI (default: 300)",
    )
    args = parser.parse_args()

    generate_visualization(args.region, args.model, args.dpi)
