#!/usr/bin/env python3
"""T0: kNN with top-K visual neighbors, re-ranked by capture year."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.baseline_infer import QUERY_SPLITS_B2, load_feature_tables, save_json_metrics
from lib.geo import haversine_km
from lib.geoclip_ops import knn_predict
from lib.metrics import evaluate, summarize
from lib.paths import META, RESULTS
from lib.time_feats import dates_to_features


def query_years(table: pd.DataFrame) -> np.ndarray:
    meta = pd.read_parquet(META / "smapshot.parquet", columns=["id", "date_shot_min", "date_shot_max"])
    merged = table.merge(meta, on="id", how="left")
    years = np.full(len(merged), np.nan, dtype=np.float64)
    for i, row in merged.iterrows():
        yr, _, has = dates_to_features(row.date_shot_min, row.date_shot_max)
        if has:
            years[i] = yr
    return years


def pick_neighbor(sim, idx, qy, train_years, gps_train, gamma: float, top_k: int):
    n = len(sim)
    pred = np.zeros((n, 2), dtype=np.float64)
    score_out = np.zeros(n, dtype=np.float64)
    nn_out = np.zeros(n, dtype=np.int64)
    cap = 50.0
    for i in range(n):
        best_j, best_s = 0, -1e9
        for j in range(min(top_k, sim.shape[1])):
            col = int(idx[i, j])
            s = float(sim[i, j])
            if np.isfinite(qy[i]) and np.isfinite(train_years[col]):
                dy = min(abs(qy[i] - train_years[col]), cap)
                s = s - gamma * (dy / cap)
            if s > best_s:
                best_j, best_s = col, s
        pred[i] = gps_train[best_j]
        score_out[i] = best_s
        nn_out[i] = col
    return pred, score_out, nn_out


def search_gamma(
    sim_v,
    idx_v,
    qy_v,
    train_years,
    gps_train,
    lat,
    lon,
    gammas,
    top_k: int,
):
    best = None
    for gamma in gammas:
        pred, _, _ = pick_neighbor(sim_v, idx_v, qy_v, train_years, gps_train, gamma, top_k)
        err = haversine_km(lat, lon, pred[:, 0], pred[:, 1])
        med = float(np.median(err))
        acc25 = float((err <= 25).mean() * 100)
        row = {"gamma": gamma, "median_km": med, "acc@25km": acc25}
        if best is None or med < best["median_km"]:
            best = row
    return best


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    top_k = 50
    pred_path = RESULTS / "preds" / "T0.parquet"
    if pred_path.is_file():
        print(f"skip {pred_path}", flush=True)
        return

    feats, _, table = load_feature_tables()
    train = table[table.split == "train"]
    queries = table[table.split.isin(QUERY_SPLITS_B2)]
    f_train = np.asarray(feats[train.feat_idx.values], dtype=np.float32)
    gps_train = train[["latitude", "longitude"]].values.astype(float)
    train_years = query_years(train)
    f_query = np.asarray(feats[queries.feat_idx.values], dtype=np.float32)
    qy = query_years(queries)

    print(f"T0 knn top-{top_k} train={len(train)} queries={len(queries)}", flush=True)
    _, sim, idx = knn_predict(f_query, f_train, gps_train, k=top_k, device=device)

    val_mask = queries.split.values == "val"
    gammas = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    best = search_gamma(
        sim[val_mask],
        idx[val_mask],
        qy[val_mask],
        train_years,
        gps_train,
        queries.latitude.values[val_mask],
        queries.longitude.values[val_mask],
        gammas,
        top_k,
    )
    gamma = float(best["gamma"])
    print(f"val best gamma={gamma} median={best['median_km']:.2f} <=25%={best['acc@25km']:.1f}", flush=True)

    pred, score, nn = pick_neighbor(sim, idx, qy, train_years, gps_train, gamma, top_k)
    out = queries[["id", "split"]].copy()
    out["lat_pred"] = pred[:, 0]
    out["lon_pred"] = pred[:, 1]
    out["knn_score"] = score
    out["neighbor_id"] = train.id.values[nn]
    out.to_parquet(pred_path, index=False)

    truth = table[["id", "latitude", "longitude", "view_type", "block"]]
    metrics, _ = evaluate(out, truth)
    save_json_metrics("T0", metrics)
    meta_out = RESULTS / "metrics" / "T0_search.json"
    meta_out.write_text(json.dumps({"top_k": top_k, "best_val": best, "gammas": gammas}, indent=2))
    val_row = metrics[(metrics.split == "val") & (metrics.view_type == "all")]
    if len(val_row):
        print("val", val_row.iloc[0][["median_km", "acc@25km"]].to_dict(), flush=True)
    print(f"wrote {pred_path}", flush=True)


if __name__ == "__main__":
    main()