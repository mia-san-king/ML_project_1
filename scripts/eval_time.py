#!/usr/bin/env python3
"""Analyse time-aware models: shuffled-date control, decade and missing-date breakdown."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from geoclip import GeoCLIP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.baseline_infer import load_feature_tables
from lib.geo import haversine_km
from lib.geoclip_ops import TimeBranch, predict_feats
from lib.metrics import summarize
from lib.paths import CKPT, GEO, META, RESULTS
from lib.time_feats import dates_to_features, load_time_feats_for_table


def load_run_val(name: str) -> dict:
    path = RESULTS / "metrics" / f"{name}_summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text()).get("val", {})


def decade_breakdown(pred: pd.DataFrame, truth: pd.DataFrame) -> list[dict]:
    rows = pred.merge(truth[["id", "latitude", "longitude", "date_shot_min", "date_shot_max"]], on="id")
    rows["err"] = haversine_km(rows.latitude, rows.longitude, rows.lat_pred, rows.lon_pred)
    buckets = []
    for _, row in rows.iterrows():
        yr, _, has = dates_to_features(row.date_shot_min, row.date_shot_max)
        buckets.append("missing" if not has else str(int(yr // 10 * 10)))
    rows["bucket"] = buckets
    out = []
    for bucket, g in rows.groupby("bucket"):
        out.append(
            {
                "bucket": bucket,
                "n": int(len(g)),
                "median_km": float(np.median(g.err)),
                "acc@25km": float((g.err <= 25).mean() * 100),
            }
        )
    return sorted(out, key=lambda r: r["bucket"])


@torch.no_grad()
def predict_with_time(run_name: str, table, feats_mmap, grid, device: str, shuffle_years: bool = False, seed: int = 0):
    model = GeoCLIP().to(device)
    model.load_state_dict(torch.load(CKPT / f"{run_name}.pt", map_location=device, weights_only=True))
    time_branch = TimeBranch().to(device)
    time_branch.load_state_dict(torch.load(CKPT / f"{run_name}_time.pt", map_location=device, weights_only=True))
    model.eval()
    time_branch.eval()

    sub = table[table.split.isin({"val", "test_in", "test_city", "test_region"})].copy()
    f = np.asarray(feats_mmap[sub.feat_idx.values], dtype=np.float32)
    tfeat = load_time_feats_for_table(sub, feats_mmap).copy()
    if shuffle_years:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(tfeat))
        tfeat[:, :11] = tfeat[perm, :11]
    preds, _, _, _ = predict_feats(model, f, grid, chunk=1024, time_branch=time_branch, time_feats=tfeat)
    out = sub[["id", "split"]].copy()
    out["lat_pred"] = preds[:, 0]
    out["lon_pred"] = preds[:, 1]
    return out


def split_errors(pred: pd.DataFrame, truth: pd.DataFrame, split: str) -> dict:
    sub = pred[pred.split == split].merge(truth[["id", "latitude", "longitude"]], on="id")
    err = haversine_km(sub.latitude, sub.longitude, sub.lat_pred, sub.lon_pred)
    return {"n": int(len(err)), "median_km": float(np.median(err)), "acc@25km": float((err <= 25).mean() * 100)}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="T2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    feats_mmap, _, table = load_feature_tables()
    grid = np.load(GEO / "grid_1km.npy")
    meta = pd.read_parquet(META / "smapshot.parquet", columns=["id", "date_shot_min", "date_shot_max"])
    truth = table.merge(meta, on="id")

    base_pred = pd.read_parquet(RESULTS / "preds" / f"{args.run_name}.parquet")
    shuf_pred = predict_with_time(args.run_name, table, feats_mmap, grid, args.device, shuffle_years=True)
    shuf_pred.to_parquet(RESULTS / "preds" / f"{args.run_name}_shuffle.parquet", index=False)

    analysis = {
        "run_name": args.run_name,
        "val_compare": {
            args.run_name: load_run_val(args.run_name),
            "A4_tau5": load_run_val("A4_tau5"),
            "T0": load_run_val("T0"),
            "B2": load_run_val("B2"),
        },
        "test_in_decade": decade_breakdown(base_pred[base_pred.split == "test_in"], truth),
        "test_region_decade": decade_breakdown(base_pred[base_pred.split == "test_region"], truth),
    }
    for split in ("test_in", "test_region"):
        analysis[f"{split}_model"] = split_errors(base_pred, truth, split)
        analysis[f"{split}_shuffle_date"] = split_errors(shuf_pred, truth, split)
        analysis[f"{split}_A4"] = split_errors(
            pd.read_parquet(RESULTS / "preds" / "A4_tau5.parquet"), truth, split
        )

    out_path = RESULTS / "metrics" / "time_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()