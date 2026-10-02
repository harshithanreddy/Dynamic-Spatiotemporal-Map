"""
Step 9: Static GCN over the hex graph. Same features and spatial block CV as the baseline,
so results are directly comparable.

Output: data/processed/<region_name>/gnn_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED, RANDOM_SEED, get_region
from src.eval.spatial_cv import spatial_block_folds
from src.models.train_baseline import FEATURE_COLS_EXCLUDE, best_f1_threshold


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


class StaticGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x).squeeze(-1)


def build_edge_index(hex_ids: list, neighbors: dict) -> torch.Tensor:
    id_to_idx = {h: i for i, h in enumerate(hex_ids)}
    src, dst = [], []
    for h, nbrs in neighbors.items():
        if h not in id_to_idx:
            continue
        for n in nbrs:
            if n in id_to_idx:
                src.append(id_to_idx[h])
                dst.append(id_to_idx[n])
    return torch.tensor([src, dst], dtype=torch.long)


def train_one_fold(data, train_mask, test_mask, pos_weight, in_channels, epochs=150, device="cpu", fold_seed=RANDOM_SEED):
    set_seed(fold_seed)
    model = StaticGCN(in_channels).to(device)
    data_dev = data.to(device)
    train_mask_dev = train_mask.to(device)
    test_mask_dev = test_mask.to(device)
    pos_weight_dev = pos_weight.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data_dev.x, data_dev.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            out[train_mask_dev], data_dev.y[train_mask_dev], pos_weight=pos_weight_dev
        )
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data_dev.x, data_dev.edge_index)
        probs = torch.sigmoid(out[test_mask_dev]).cpu().numpy()
    return probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Static GCN on region hex graph.")
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

    neighbors_path = processed_dir / "hex_neighbors.json"
    if not neighbors_path.exists():
        raise FileNotFoundError(f"Hex neighbors map not found at {neighbors_path}. Run build_h3_grid.py --region {region.name} first.")

    df = pd.read_parquet(features_path).reset_index(drop=True)
    with open(neighbors_path) as f:
        neighbors = json.load(f)

    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
    hex_ids = df["hex_id"].tolist()
    edge_index = build_edge_index(hex_ids, neighbors)
    print(f"[gnn] {region.name.upper()} Graph: {len(hex_ids)} nodes, {edge_index.shape[1]} directed edges")

    X_all = StandardScaler().fit_transform(df[feature_cols].values)
    x = torch.tensor(X_all, dtype=torch.float)
    y = torch.tensor(df["label"].values, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)

    pos_weight = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gnn] Using device: {device}")

    aucs, f1s = [], []
    oof_probs = np.full(len(df), np.nan, dtype=float)
    n_blocks = args.n_blocks_per_side if args.n_blocks_per_side is not None else region.default_n_blocks_per_side
    for fold_i, (train_idx, test_idx) in enumerate(spatial_block_folds(df, n_folds=args.n_folds, n_blocks_per_side=n_blocks, seed=RANDOM_SEED)):
        train_mask = torch.zeros(len(df), dtype=torch.bool)
        test_mask = torch.zeros(len(df), dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        probs = train_one_fold(
            data, train_mask, test_mask, pos_weight, in_channels=len(feature_cols), device=device, fold_seed=RANDOM_SEED + fold_i
        )
        oof_probs[test_idx] = probs
        y_test = df["label"].iloc[test_idx].values

        if df["label"].iloc[test_idx].nunique() < 2:
            print(f"[gnn] Fold {fold_i}: skipping AUC metric (only 1 class in test set)")
            continue

        auc = roc_auc_score(y_test, probs)
        threshold, f1 = best_f1_threshold(y_test, probs)
        aucs.append(auc)
        f1s.append(f1)
        print(f"[gnn] Fold {fold_i}: AUC {auc:.3f}")

    valid_mask = ~np.isnan(oof_probs)
    y_true_all = df["label"].values
    oof_threshold, _ = best_f1_threshold(y_true_all[valid_mask], oof_probs[valid_mask])
    y_pred_label = np.zeros(len(df), dtype=int)
    y_pred_label[valid_mask] = (oof_probs[valid_mask] >= oof_threshold).astype(int)

    pred_df = pd.DataFrame({
        "h3_index": df["hex_id"],
        "region": region.name,
        "y_true": y_true_all,
        "y_pred_prob": oof_probs,
        "y_pred_label": y_pred_label,
    })
    out_pred = processed_dir / "predictions_gcn.csv"
    pred_df.to_csv(out_pred, index=False)
    print(f"[gnn] Saved predictions -> {out_pred} (F1 threshold: {oof_threshold:.3f})")

    results = {
        "static_gcn": {
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "f1_mean": float(np.mean(f1s)),
            "n_folds_used": len(aucs),
        }
    }
    out_json = processed_dir / "gnn_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(
        f"[gnn] Static GCN ({region.name.upper()}): AUC {results['static_gcn']['auc_mean']:.3f} "
        f"+/- {results['static_gcn']['auc_std']:.3f}"
    )
    print(f"[gnn] Saved -> {out_json}")

    # Quick comparison against baseline, if it's already been run for this region
    baseline_path = processed_dir / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"\n[compare] AUC-ROC summary ({region.name.upper()}):")
        print(f"  Logistic regression : {baseline['logistic_regression']['auc_mean']:.3f}")
        print(f"  Random forest        : {baseline['random_forest']['auc_mean']:.3f}")
        print(f"  Static GCN           : {results['static_gcn']['auc_mean']:.3f}")
