"""
Script to fetch full USGS earthquake catalog for Noto Peninsula via USGS FDSN Web API.
"""
import sys
from pathlib import Path
import urllib.request
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW, get_region


def fetch_usgs_catalog():
    # Derive bounding box from H3 grid extent in data/noto/processed/noto_features.parquet
    # grid extent: lon [136.671, 137.358], lat [36.719, 37.530]
    bbox = {
        "min_lon": 136.60,
        "max_lon": 137.40,
        "min_lat": 36.65,
        "max_lat": 37.55,
    }
    print(f"[usgs] Bounding box derived from H3 grid extent: {bbox}")

    # Query USGS FDSN API
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "csv",
        "starttime": "2024-01-01T00:00:00",
        "endtime": "2025-07-30T23:59:59",
        "minmagnitude": "2.4",
        "minlatitude": str(bbox["min_lat"]),
        "maxlatitude": str(bbox["max_lat"]),
        "minlongitude": str(bbox["min_lon"]),
        "maxlongitude": str(bbox["max_lon"]),
        "orderby": "time-asc",
    }

    query_str = "&".join(f"{k}={v}" for k, v in params.items())
    full_url = f"{base_url}?{query_str}"
    print(f"[usgs] Requesting: {full_url}")

    req = urllib.request.Request(full_url, headers={"User-Agent": "Antigravity/1.0"})
    out_dir = RAW / "noto" / "earthquakes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "noto_usgs_catalog_full.csv"

    with urllib.request.urlopen(req) as resp:
        content = resp.read()

    with open(out_file, "wb") as f:
        f.write(content)

    df = pd.read_csv(out_file)
    print(f"[usgs] Saved {len(df)} records to -> {out_file}")

    # Verify pagination / limits
    if len(df) >= 20000:
        print("[warn] USGS API returned 20,000 records — limit reached! Pagination required.")
    else:
        print(f"[usgs] API limit is 20,000. Total fetched = {len(df)} (< 20,000), so full dataset was retrieved in 1 page.")

    return df


if __name__ == "__main__":
    fetch_usgs_catalog()
