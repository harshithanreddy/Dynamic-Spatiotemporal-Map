"""
Snapshot Timeline Construction for Dynamic Feature Layer.

Loads earthquake catalog for a region and builds a snapshot timeline using a hybrid trigger rule:
1. Event-triggered snapshot whenever a catalog event >= min_mag occurs.
2. Regular fallback snapshot every regular_interval_hours during quiet gaps.
3. Collapse rule: if a regular-bin snapshot falls within collapse_window_hours of an event snapshot,
   keep only the event-triggered one.

Output: data/processed/<region_name>/dynamic/snapshot_timeline.csv
"""
import argparse
from datetime import timedelta
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED, get_region


def build_timeline(
    catalog_path: Path,
    min_mag: float = 3.5,
    regular_interval_hours: float = 24.0,
    collapse_window_hours: float = 1.0,
) -> pd.DataFrame:
    """
    Build hybrid snapshot timeline from earthquake catalog.
    """
    if not catalog_path.exists():
        raise FileNotFoundError(f"Earthquake catalog not found at {catalog_path}")

    catalog = pd.read_csv(catalog_path)
    if "time" not in catalog.columns:
        raise ValueError(f"Catalog at {catalog_path} missing 'time' column")

    catalog["time_dt"] = pd.to_datetime(catalog["time"], utc=True)
    catalog = catalog.sort_values("time_dt").reset_index(drop=True)

    # 1. Event candidates (mag >= min_mag)
    events = catalog[catalog["mag"] >= min_mag].copy()

    t_start = catalog["time_dt"].min().floor("D")
    t_end = catalog["time_dt"].max().ceil("D")

    # 2. Regular candidates
    freq_str = f"{int(regular_interval_hours)}h"
    regular_times = pd.date_range(start=t_start, end=t_end, freq=freq_str, tz="UTC")

    # 3. Collapse rule: drop regular snapshot if within collapse_window_hours of an event snapshot
    event_times = events["time_dt"].tolist()
    collapse_td = timedelta(hours=collapse_window_hours)

    filtered_regular_times = []
    for r_time in regular_times:
        close_to_event = any(abs(r_time - e_time) <= collapse_td for e_time in event_times)
        if not close_to_event:
            filtered_regular_times.append(r_time)

    # 4. Construct output dataframe
    rows = []
    for idx, row in events.iterrows():
        rows.append(
            {
                "timestamp": row["time_dt"],
                "trigger_type": "event",
                "mag": float(row["mag"]),
                "latitude": float(row["latitude"]) if "latitude" in row else np.nan,
                "longitude": float(row["longitude"]) if "longitude" in row else np.nan,
                "depth": float(row["depth"]) if "depth" in row else np.nan,
                "event_id": str(row["id"]) if "id" in row else f"event_{idx}",
            }
        )

    for r_time in filtered_regular_times:
        rows.append(
            {
                "timestamp": r_time,
                "trigger_type": "regular",
                "mag": np.nan,
                "latitude": np.nan,
                "longitude": np.nan,
                "depth": np.nan,
                "event_id": None,
            }
        )

    df_snapshots = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df_snapshots["snapshot_id"] = [f"snap_{i:04d}" for i in range(len(df_snapshots))]

    # Reorder columns
    cols = ["snapshot_id", "timestamp", "trigger_type", "mag", "latitude", "longitude", "depth", "event_id"]
    return df_snapshots[cols]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build hybrid snapshot timeline for dynamic features.")
    parser.add_argument("--region", type=str, choices=["noto", "hokkaido"], default="noto", help="Region name (default: noto)")
    parser.add_argument("--min-mag", type=float, default=3.5, help="Minimum magnitude threshold for event snapshots (default: 3.5)")
    parser.add_argument("--regular-interval-hours", type=float, default=24.0, help="Interval in hours for fallback regular snapshots (default: 24.0)")
    parser.add_argument("--collapse-window-hours", type=float, default=1.0, help="Window in hours to collapse regular snapshot if near event (default: 1.0)")
    args = parser.parse_args()

    region = get_region(args.region)
    dynamic_dir = PROCESSED / region.name / "dynamic"
    dynamic_dir.mkdir(parents=True, exist_ok=True)

    df_timeline = build_timeline(
        catalog_path=region.earthquake_catalog,
        min_mag=args.min_mag,
        regular_interval_hours=args.regular_interval_hours,
        collapse_window_hours=args.collapse_window_hours,
    )

    out_csv = dynamic_dir / "snapshot_timeline.csv"
    df_timeline.to_csv(out_csv, index=False)

    print("==================================================")
    print(f"SNAPSHOT TIMELINE GENERATED ({region.name.upper()})")
    print("==================================================")
    print(f"Catalog source      : {region.earthquake_catalog}")
    print(f"Min magnitude       : {args.min_mag}")
    print(f"Regular interval    : {args.regular_interval_hours} hours")
    print(f"Collapse window     : {args.collapse_window_hours} hours")
    print(f"Total Snapshot Count: {len(df_timeline)}")
    print("\nTrigger Breakdown:")
    breakdown = df_timeline["trigger_type"].value_counts()
    for trigger_type, count in breakdown.items():
        pct = (count / len(df_timeline)) * 100
        print(f"  {trigger_type:<10}: {count:4d} ({pct:5.1f}%)")
    print(f"\nSaved timeline to -> {out_csv}")
    print("==================================================")
