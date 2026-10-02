"""
Spatial block cross-validation: split hexes into a coarse grid of blocks, assign whole blocks to
folds (not individual hexes) so that train/test hexes are never adjacent. Prevents the spatial
autocorrelation leakage that random k-fold would allow.
"""
import numpy as np
import pandas as pd


def make_spatial_blocks(df: pd.DataFrame, n_blocks_per_side: int = 6) -> pd.Series:
    """Assign each row to a block based on a coarse lon/lat grid over the study area."""
    lon_bins = pd.cut(df["lon"], bins=n_blocks_per_side, labels=False)
    lat_bins = pd.cut(df["lat"], bins=n_blocks_per_side, labels=False)
    return lon_bins * n_blocks_per_side + lat_bins


def spatial_block_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    n_blocks_per_side: int = 6,
    seed: int = 42,
    balance_labels: bool = True,
):
    """
    Yields (train_idx, test_idx) numpy index arrays, n_folds times, with whole spatial blocks
    assigned to each fold.

    When balance_labels=True and 'label' is in df, spatial blocks are greedily assigned to folds
    by a combined score balancing both cumulative positive label counts and total fold size.
    """
    blocks = make_spatial_blocks(df, n_blocks_per_side)

    if "label" in df.columns and balance_labels:
        block_pos = df.groupby(blocks)["label"].sum()
        block_size = df.groupby(blocks).size()
    else:
        block_pos = df.groupby(blocks).size()
        block_size = block_pos

    unique_blocks = list(block_pos.index)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_blocks)

    # Sort blocks by descending positive count (secondary key: block size descending)
    unique_blocks.sort(key=lambda b: (block_pos[b], block_size[b]), reverse=True)

    total_pos = sum(block_pos.values)
    total_size = len(df)

    fold_pos_totals = [0] * n_folds
    fold_size_totals = [0] * n_folds
    fold_assignment = {}

    for block in unique_blocks:
        b_pos = block_pos[block]
        b_size = block_size[block]

        # Score each fold using a combined normalized metric balancing positives and total size
        scores = [
            (fold_pos_totals[f] / max(total_pos, 1)) + (fold_size_totals[f] / total_size)
            for f in range(n_folds)
        ]
        min_fold = int(np.argmin(scores))
        fold_assignment[block] = min_fold
        fold_pos_totals[min_fold] += b_pos
        fold_size_totals[min_fold] += b_size

    block_fold = blocks.map(fold_assignment)

    for fold in range(n_folds):
        test_idx = df.index[block_fold == fold].to_numpy()
        train_idx = df.index[block_fold != fold].to_numpy()
        yield train_idx, test_idx
