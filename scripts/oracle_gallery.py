#!/usr/bin/env python3
"""Oracle error: nearest point in a candidate gallery to each ground-truth GPS."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.geo import R_EARTH
from lib.paths import GEO, META, RESULTS


def oracle_errors(truth_lat, truth_lon, gallery):
    gallery = np.asarray(gallery, dtype=float)
    truth = np.deg2rad(np.column_stack([truth_lat, truth_lon]))
    tree = BallTree(np.deg2rad(gallery), metric="haversine")
    dist_rad, _ = tree.query(truth, k=1)
    return dist_rad[:, 0] * R_EARTH


def main():
    meta = pd.read_parquet(META / "smapshot.parquet", columns=["id", "latitude", "longitude"])
    splits = pd.read_csv("data/splits/main.csv.gz")
    table = splits.merge(meta, on="id")
    table = table[~table.split.str.startswith("nadir_")]
    val = table[table.split == "val"].dropna(subset=["latitude", "longitude"])

    grid_path = GEO / "grid_1km.npy"
    grid = np.load(grid_path)
    train_coords = (
        table[table.split == "train"][["latitude", "longitude"]].drop_duplicates().to_numpy(dtype=float)
    )

    out = {}
    for name, gallery in [("grid_1km", grid), ("train_unique", train_coords)]:
        errors = oracle_errors(val.latitude.values, val.longitude.values, gallery)
        out[name] = {
            "n_val": int(len(val)),
            "n_gallery": int(len(gallery)),
            "median_km": float(np.median(errors)),
            "p90_km": float(np.percentile(errors, 90)),
            "acc@25km": float((errors <= 25).mean() * 100),
        }
        print(name, out[name])

    dest = RESULTS / "metrics" / "oracle_val.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()