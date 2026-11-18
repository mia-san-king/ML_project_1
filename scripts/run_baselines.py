#!/usr/bin/env python3
"""B1/B2/B3 baselines on cached CLIP features (all non-nadir splits; skip existing outputs)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from geoclip import GeoCLIP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.baseline_infer import (
    ALL_EVAL_SPLITS,
    QUERY_SPLITS_B2,
    ensure_gallery_emb,
    ensure_mlp512,
    load_feature_tables,
    run_geoclip_baseline,
    save_json_metrics,
)
from lib.geoclip_ops import knn_predict
from lib.metrics import evaluate
from lib.paths import GEO, RESULTS


def run_b2(table, feats, device: str):
    pred_path = RESULTS / "preds" / "B2.parquet"
    if pred_path.is_file():
        print("skip B2 (outputs exist)", flush=True)
        return

    train = table[table.split == "train"]
    queries = table[table.split.isin(QUERY_SPLITS_B2)]
    f_train = np.asarray(feats[train.feat_idx.values], dtype=np.float32)
    gps_train = train[["latitude", "longitude"]].values
    f_query = np.asarray(feats[queries.feat_idx.values], dtype=np.float32)

    print(f"B2 knn train={len(train)} queries={len(queries)}", flush=True)
    gps_pred, sim, nn_idx = knn_predict(f_query, f_train, gps_train, k=1, device=device)
    neighbor_ids = train.id.values[nn_idx[:, 0]]

    out = queries[["id", "split"]].copy()
    out["lat_pred"] = gps_pred[:, 0]
    out["lon_pred"] = gps_pred[:, 1]
    out["knn_sim"] = sim[:, 0]
    out["neighbor_id"] = neighbor_ids
    out.to_parquet(pred_path, index=False)

    truth = table[["id", "latitude", "longitude", "view_type", "block"]]
    metrics, _ = evaluate(out, truth)
    save_json_metrics("B2", metrics)
    print(f"wrote {pred_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=("b1", "b2", "b3", "all"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    feats, ids, table = load_feature_tables()
    which = {args.which} if args.which != "all" else {"b1", "b2", "b3"}

    model = None
    if which & {"b1", "b3"}:
        model = GeoCLIP().to(args.device).eval()
        mlp512 = ensure_mlp512(model, feats, ids)
    else:
        mlp512 = None

    if "b1" in which:
        gallery = model.gps_gallery.detach().cpu().numpy()
        ensure_gallery_emb(model, gallery, "mp16")
        run_geoclip_baseline(model, table, mlp512, gallery, "mp16", "B1", args.device)

    if "b3" in which:
        grid = np.load(GEO / "grid_1km.npy")
        run_geoclip_baseline(model, table, mlp512, grid, "grid1km", "B3", args.device)

    if "b2" in which:
        run_b2(table, feats, args.device)


if __name__ == "__main__":
    main()