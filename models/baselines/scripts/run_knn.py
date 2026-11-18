#!/usr/bin/env python3
"""B2: nearest training-image coordinates using cached CLIP feature similarity."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib.feature_tables import QUERY_SPLITS_B2, load_feature_tables, save_json_metrics
from lib.metrics import evaluate
from lib.paths import RESULTS
from models.baselines.lib.knn import knn_predict


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
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pred_path, index=False)

    truth = table[["id", "latitude", "longitude", "view_type", "block"]]
    metrics, _ = evaluate(out, truth)
    save_json_metrics("B2", metrics)
    print(f"wrote {pred_path}", flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    feats, _, table = load_feature_tables()
    run_b2(table, feats, args.device)


if __name__ == "__main__":
    main()
