"""Shared data loading and GeoCLIP retrieval baselines (cached mlp512 + gallery emb)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lib.geoclip_ops import compute_mlp512, embed_gallery, predict_mlp512
from lib.metrics import evaluate, summarize
from lib.geo import haversine_km
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


def ensure_mlp512(model, feats, ids) -> np.ndarray:
    out = FEATS / "clip_mlp512.npy"
    if out.is_file():
        arr = np.load(out, mmap_mode="r")
        if arr.shape[0] == len(ids):
            return arr
    print("computing clip_mlp512...", flush=True)
    mlp512 = compute_mlp512(model, feats)
    np.save(out, mlp512)
    print(f"wrote {out} shape={mlp512.shape}", flush=True)
    return mlp512


def load_gallery_emb(gallery_name: str, device: str) -> tuple[np.ndarray, torch.Tensor]:
    gps = np.load(FEATS / f"gallery_{gallery_name}_gps.npy")
    emb = torch.from_numpy(np.load(FEATS / f"gallery_{gallery_name}_emb.npy").astype(np.float32)).to(device)
    return gps, emb


def ensure_gallery_emb(model, gallery: np.ndarray, gallery_name: str) -> tuple[np.ndarray, torch.Tensor]:
    gps_path = FEATS / f"gallery_{gallery_name}_gps.npy"
    emb_path = FEATS / f"gallery_{gallery_name}_emb.npy"
    gps = np.asarray(gallery, dtype=np.float32)
    if not gps_path.is_file():
        np.save(gps_path, gps)
    if not emb_path.is_file():
        print(f"embedding gallery {gallery_name} n={len(gps)}...", flush=True)
        emb = embed_gallery(model, gps).half().cpu().numpy()
        np.save(emb_path, emb)
    return load_gallery_emb(gallery_name, str(model.logit_scale.device))


@torch.no_grad()
def predict_splits(
    model,
    table: pd.DataFrame,
    mlp512,
    gallery_name: str,
    *,
    split_names: set[str],
    device: str,
    penalty=None,
    lam: float = 0.0,
    top_k: int = 20,
):
    gps, gallery_emb = load_gallery_emb(gallery_name, device)
    sub = table[table.split.isin(split_names)].copy()
    mlp_q = np.asarray(mlp512[sub.feat_idx.values], dtype=np.float32)
    preds, top_idx, top_log, lse = predict_mlp512(
        model,
        mlp_q,
        gps,
        gallery_emb=gallery_emb,
        penalty=penalty,
        lam=lam,
        top_k=top_k,
    )
    return sub, preds, top_idx, top_log, lse, gps


def val_metrics_from_preds(sub: pd.DataFrame, preds_gps: np.ndarray) -> dict:
    err = haversine_km(
        sub.latitude.values,
        sub.longitude.values,
        preds_gps[:, 0],
        preds_gps[:, 1],
    )
    return summarize(err)


def run_geoclip_baseline(
    model,
    table,
    mlp512,
    gallery,
    gallery_name: str,
    pred_name: str,
    device: str,
    *,
    penalty=None,
    lam: float = 0.0,
    skip_if_exists: bool = True,
):
    pred_path = RESULTS / "preds" / f"{pred_name}.parquet"
    topk_path = RESULTS / "preds" / f"{pred_name}_topk.npz"
    if skip_if_exists and pred_path.is_file() and topk_path.is_file():
        print(f"skip {pred_name} (outputs exist)", flush=True)
        return

    if not (FEATS / f"gallery_{gallery_name}_emb.npy").is_file():
        ensure_gallery_emb(model, gallery, gallery_name)

    sub, preds_gps, top_idx, top_log, lse, _ = predict_splits(
        model,
        table,
        mlp512,
        gallery_name,
        split_names=ALL_EVAL_SPLITS,
        device=device,
        penalty=penalty,
        lam=lam,
    )

    out = sub[["id", "split"]].copy()
    out["lat_pred"] = preds_gps[:, 0]
    out["lon_pred"] = preds_gps[:, 1]
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pred_path, index=False)

    np.savez_compressed(
        topk_path,
        ids=sub.id.values.astype(np.int64),
        split=sub.split.values,
        top_idx=top_idx.astype(np.int32),
        top_logit=top_log.astype(np.float32),
        logsumexp=lse.astype(np.float32),
        gallery_name=np.array(gallery_name),
        lam=np.float32(lam),
        regional_penalty=np.array(penalty is not None),
    )
    truth = table[["id", "latitude", "longitude", "view_type", "block"]]
    metrics, _ = evaluate(out, truth)
    save_json_metrics(pred_name, metrics)
    print(f"wrote {pred_path} {topk_path}", flush=True)