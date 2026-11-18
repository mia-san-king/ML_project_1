"""Shared cached-feature tables and metric JSON output."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.paths import FEATS, META, RESULTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = PROJECT_ROOT / "data/splits/main.csv.gz"

QUERY_SPLITS_B2 = {"val", "test_in", "test_city", "test_region"}
ALL_EVAL_SPLITS = {"train", "val", "test_in", "test_city", "test_region"}


def load_feature_tables():
    feats = np.load(FEATS / "clip_vitl14.npy", mmap_mode="r")
    ids = np.load(FEATS / "clip_vitl14_ids.npy")
    id_to_idx = {int(i): k for k, i in enumerate(ids)}
    splits = pd.read_csv(SPLIT_CSV)
    meta = pd.read_parquet(META / "smapshot.parquet", columns=["id", "latitude", "longitude", "view_type"])
    table = splits.merge(meta, on="id", how="inner")
    table = table[~table.split.str.startswith("nadir_")]
    table = table[table.id.isin(id_to_idx)]
    table["feat_idx"] = table.id.map(id_to_idx)
    return feats, ids, table


def save_json_metrics(name: str, metrics: pd.DataFrame) -> None:
    path = RESULTS / "metrics" / f"{name}_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    val_row = metrics[(metrics.split == "val") & (metrics.view_type == "all")]
    payload = {
        "splits": metrics.to_dict(orient="records"),
        "val": val_row.iloc[0].to_dict() if len(val_row) else {},
    }
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {path}", flush=True)
