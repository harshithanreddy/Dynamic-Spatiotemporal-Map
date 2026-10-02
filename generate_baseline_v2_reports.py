import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Compute canonical OOF metrics directly from saved prediction CSVs
v2_data = []
for reg in ["noto", "hokkaido"]:
    n_blocks = 3 if reg == "noto" else 8
    for model in ["logistic_regression", "random_forest", "gcn"]:
        pred_path = Path(f"data/processed/{reg}/predictions_{model}.csv")
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions file at {pred_path}. Train models first.")
        df = pd.read_csv(pred_path)
        valid = df["y_pred_prob"].notna()
        y_true = df.loc[valid, "y_true"].values
        y_prob = df.loc[valid, "y_pred_prob"].values
        y_pred = df.loc[valid, "y_pred_label"].values

        auc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)

        v2_data.append({
            "region": reg,
            "model": model,
            "n_blocks_per_side": n_blocks,
            "auc_mean": float(auc),
            "auc_std": 0.0,
            "f1_score": float(f1),
            "n_folds": 5,
        })

df_v2 = pd.DataFrame(v2_data)
v2_path = OUTPUTS_DIR / "baseline_results_v2.csv"
df_v2.to_csv(v2_path, index=False)
print(f"[report] Saved -> {v2_path}")

# Historical comparison data (Original block size n=6 vs De-leaked n=3 Noto / n=8 Hokkaido)
old_metrics = {
    ("noto", "logistic_regression"): {"old_block_size": 6, "old_auc": 0.927, "old_f1": 0.690},
    ("noto", "random_forest"): {"old_block_size": 6, "old_auc": 0.943, "old_f1": 0.720},
    ("noto", "gcn"): {"old_block_size": 6, "old_auc": 0.897, "old_f1": 0.670},
    ("hokkaido", "logistic_regression"): {"old_block_size": 6, "old_auc": 0.906, "old_f1": 0.320},
    ("hokkaido", "random_forest"): {"old_block_size": 6, "old_auc": 0.934, "old_f1": 0.450},
    ("hokkaido", "gcn"): {"old_block_size": 6, "old_auc": 0.949, "old_f1": 0.510},
}

comparison_data = []
for row in v2_data:
    key = (row["region"], row["model"])
    old = old_metrics[key]
    new_auc = row["auc_mean"]
    new_f1 = row["f1_score"]
    delta = new_auc - old["old_auc"]

    comparison_data.append({
        "region": row["region"],
        "model": row["model"],
        "old_block_size": old["old_block_size"],
        "old_auc": old["old_auc"],
        "new_block_size": row["n_blocks_per_side"],
        "new_auc": float(new_auc),
        "auc_delta": float(delta),
        "old_f1": old["old_f1"],
        "new_f1": float(new_f1),
    })

df_comp = pd.DataFrame(comparison_data)
comp_path = OUTPUTS_DIR / "baseline_releak_comparison.csv"
df_comp.to_csv(comp_path, index=False)
print(f"[report] Saved -> {comp_path}\n")

# Print Summary Table
print("=" * 80)
print("RE-BASELINED MODEL PERFORMANCE COMPARISON (BEFORE vs AFTER DE-LEAKING)")
print("=" * 80)
print(f"{'Region':<10} | {'Model':<20} | {'Old (n, AUC, F1)':<20} | {'New (n, AUC, F1)':<20} | {'AUC Delta':<10}")
print("-" * 80)
for row in comparison_data:
    old_str = f"n={row['old_block_size']}, {row['old_auc']:.3f}, {row['old_f1']:.3f}"
    new_str = f"n={row['new_block_size']}, {row['new_auc']:.3f}, {row['new_f1']:.3f}"
    delta_str = f"{row['auc_delta']:+.3f}"
    print(f"{row['region'].upper():<10} | {row['model']:<20} | {old_str:<20} | {new_str:<20} | {delta_str:<10}")
print("=" * 80)

# Mandatory Caveat Notice
caveat_text = (
    "Hokkaido's re-baselined AUC still reflects a block size (n=8) on the steep part of the "
    "leakage curve, not a fully de-leaked estimate — this is a fold-count constraint, not a choice, "
    "and should be reported as a named limitation rather than presented as a solved problem."
)

print("\n" + "!" * 80)
print("MANDATORY LIMITATION CAVEAT:")
print(caveat_text)
print("!" * 80 + "\n")
