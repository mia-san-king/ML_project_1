"""Capture-date features aligned with cached CLIP rows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.paths import FEATS, META

YEAR_PERIODS = (2.0, 5.0, 10.0, 25.0, 50.0)
YEAR_ANCHOR = 1960.0
YEAR_SCALE = 40.0
TIME_DIM = 1 + 2 * len(YEAR_PERIODS) + 2  # year_norm, sin/cos, log_span, has_date


def build_time_feats_table(meta: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    sub = meta.set_index("id").reindex(ids.astype(np.int64))
    a = pd.to_datetime(sub["date_shot_min"], errors="coerce", utc=True)
    b = pd.to_datetime(sub["date_shot_max"], errors="coerce", utc=True)
    mid = a + (b - a) / 2
    year = mid.dt.year + (mid.dt.dayofyear - 1) / 366.0
    span = (b - a).dt.days.clip(lower=0).fillna(0)
    has = a.notna() & b.notna() & year.between(1800, 2030)
    year = year.where(has, 0.0).astype(np.float64)

    out = np.zeros((len(ids), TIME_DIM), dtype=np.float32)
    ynorm = ((year - YEAR_ANCHOR) / YEAR_SCALE).astype(np.float32)
    out[:, 0] = np.where(has, ynorm, 0.0)
    col = 1
    for period in YEAR_PERIODS:
        phase = 2 * np.pi * year / period
        out[:, col] = np.where(has, np.sin(phase), 0.0)
        out[:, col + 1] = np.where(has, np.cos(phase), 0.0)
        col += 2
    out[:, col] = np.where(has, np.log1p(span).astype(float), 0.0)
    out[:, col + 1] = has.astype(np.float32)
    return out


def ensure_time_feats() -> np.ndarray:
    """Write $P2_DATA/features/time_feats.npy aligned with clip_vitl14_ids."""
    ids_path = FEATS / "clip_vitl14_ids.npy"
    out_path = FEATS / "time_feats.npy"
    if out_path.is_file():
        arr = np.load(out_path, mmap_mode="r")
        ids = np.load(ids_path)
        if arr.shape == (len(ids), TIME_DIM):
            return arr
    ids = np.load(ids_path)
    meta = pd.read_parquet(META / "smapshot.parquet", columns=["id", "date_shot_min", "date_shot_max"])
    arr = build_time_feats_table(meta, ids)
    np.save(out_path, arr)
    print(f"wrote {out_path} shape={arr.shape}", flush=True)
    return arr


def load_time_feats_for_table(table: pd.DataFrame, feats_mmap: np.ndarray) -> np.ndarray:
    ensure_time_feats()
    full = np.load(FEATS / "time_feats.npy", mmap_mode="r")
    return np.asarray(full[table.feat_idx.values], dtype=np.float32)


def dates_to_features(date_min, date_max) -> tuple[float, float, int]:
    """Scalar helper for scripts (single row)."""
    a = pd.to_datetime(date_min, errors="coerce", utc=True)
    b = pd.to_datetime(date_max, errors="coerce", utc=True)
    if pd.isna(a) or pd.isna(b):
        return 0.0, 0.0, 0
    mid = a + (b - a) / 2
    year = float(mid.year) + (float(mid.dayofyear) - 1) / 366.0
    span = max((b - a).days, 0)
    if not (1800 <= year <= 2030):
        return 0.0, 0.0, 0
    return year, float(np.log1p(span)), 1