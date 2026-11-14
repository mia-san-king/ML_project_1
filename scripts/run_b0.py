#!/usr/bin/env python3
"""B0: constant prediction at LV95 median of non-nadir train coordinates."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pyproj import Transformer

from lib.geo import to_lv95
from lib.metrics import evaluate
from lib.paths import META, RESULTS

TO_WGS = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)


def lv95_median_to_latlon(easting, northing):
    lon, lat = TO_WGS.transform(float(easting), float(northing))
    return lat, lon


def main():
    meta = pd.read_parquet(META / "smapshot.parquet")
    splits = pd.read_csv("data/splits/main.csv.gz")
    table = splits.merge(meta[["id", "latitude", "longitude", "view_type"]], on="id")
    table = table[~table.split.str.startswith("nadir_")]
    train = table[table.split == "train"].dropna(subset=["latitude", "longitude"])

    easting, northing = to_lv95(train.latitude.values, train.longitude.values)
    lat_pred, lon_pred = lv95_median_to_latlon(np.median(easting), np.median(northing))
    print(f"B0 constant lat={lat_pred:.6f} lon={lon_pred:.6f} from n_train={len(train)}")

    preds = table[["id", "split"]].copy()
    preds["lat_pred"] = lat_pred
    preds["lon_pred"] = lon_pred

    out_dir = RESULTS / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "B0.parquet"
    preds.to_parquet(pred_path, index=False)

    truth = table[["id", "latitude", "longitude", "view_type", "block"]]
    metrics, detail = evaluate(preds, truth)
    val = metrics[(metrics.split == "val") & (metrics.view_type == "all")].iloc[0]

    metrics_path = RESULTS / "metrics" / "B0_summary.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "constant_lat": lat_pred,
        "constant_lon": lon_pred,
        "n_train_for_median": int(len(train)),
        "val": {
            "n": int(val.n),
            "median_km": float(val.median_km),
            "acc@25km": float(val["acc@25km"]),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {pred_path}")
    print(f"wrote {metrics_path}")
    print(payload["val"])


if __name__ == "__main__":
    main()