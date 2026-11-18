#!/usr/bin/env python3
"""Fine-tune GeoCLIP on cached train features; eval on Swiss 1 km grid."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from geoclip import GeoCLIP

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib.feature_tables import ALL_EVAL_SPLITS, load_feature_tables, save_json_metrics
from lib.geo import haversine_km
from models.geoclip.lib.geoclip_ops import predict_feats
from models.geoclip.lib.geoclip_ops import train as geoclip_train
from lib.paths import CKPT, GEO, RESULTS

VAL_SUBSET = 5000


def load_feature_paths(name: str = "clip_vitl14"):
    from lib.paths import FEATS

    feat_path = FEATS / f"{name}.npy"
    ids_path = FEATS / f"{name}_ids.npy"
    if not feat_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"missing {feat_path} or {ids_path}; run S6 first")
    return feat_path, ids_path


def build_neg_pool(model, cfg, device: str):
    mode = cfg.get("negatives", "swiss_grid")
    if mode == "swiss_grid":
        return torch.as_tensor(np.load(GEO / "grid_1km.npy"), dtype=torch.float32, device=device)
    if mode == "global_queue":
        return model.gps_gallery.to(device)
    raise ValueError(f"unknown negatives={mode}")


def val_subset_indices(table: pd.DataFrame, seed: int) -> np.ndarray:
    val = table[table.split == "val"]
    n = min(VAL_SUBSET, len(val))
    rng = np.random.default_rng(seed)
    return rng.choice(val.feat_idx.values.astype(np.int64), size=n, replace=False)


def make_val_fn(model, feats_mmap, val_idx: np.ndarray, val_truth: pd.DataFrame, grid, device: str):
    grid = np.asarray(grid, dtype=float)
    feats = np.asarray(feats_mmap[val_idx], dtype=np.float32)
    lat = val_truth.latitude.values
    lon = val_truth.longitude.values

    def val_fn(m):
        m.eval()
        with torch.no_grad():
            preds, _, _, _ = predict_feats(m, feats, grid, top_k=1, chunk=256)
        err = haversine_km(lat, lon, preds[:, 0], preds[:, 1])
        return float(np.median(err))

    return val_fn


@torch.no_grad()
def export_predictions(model, table, feats_mmap, grid, run_name: str):
    pred_path = RESULTS / "preds" / f"{run_name}.parquet"
    if pred_path.is_file():
        print(f"skip export {run_name} (exists)", flush=True)
        return
    sub = table[table.split.isin(ALL_EVAL_SPLITS)].copy()
    order = sub.feat_idx.values
    f = np.asarray(feats_mmap[order], dtype=np.float32)
    preds, _, _, _ = predict_feats(model, f, np.asarray(grid, dtype=float), top_k=1, chunk=1024)
    out = sub[["id", "split"]].copy()
    out["lat_pred"] = preds[:, 0]
    out["lon_pred"] = preds[:, 1]
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pred_path, index=False)
    truth = table[["id", "latitude", "longitude", "view_type", "block"]]
    from lib.metrics import evaluate

    metrics, _ = evaluate(out, truth)
    save_json_metrics(run_name, metrics)
    log_path = RESULTS / "metrics" / f"{run_name}_train_log.json"
    print(f"wrote {pred_path}", flush=True)


def run_training(cfg: dict, config_path: Path, device: str):
    run_name = cfg.get("run_name", config_path.stem)
    feat_name = cfg.get("features", "clip_vitl14")
    feat_path, _ = load_feature_paths(feat_name)
    feats_mmap = np.load(feat_path, mmap_mode="r")

    _, _, table = load_feature_tables()
    train_df = table[table.split == "train"]
    if len(train_df) == 0:
        raise RuntimeError("no train rows")

    grid = np.load(GEO / "grid_1km.npy")
    val_idx = val_subset_indices(table, int(cfg.get("seed", 0)))
    val_rows = table[table.feat_idx.isin(val_idx)]

    ckpt_path = CKPT / f"{run_name}.pt"
    model = GeoCLIP().to(device)

    if ckpt_path.is_file():
        print(f"load checkpoint {ckpt_path}", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        f_tr = torch.as_tensor(np.asarray(feats_mmap[train_df.feat_idx.values], dtype=np.float32), device=device)
        g_tr = torch.as_tensor(train_df[["latitude", "longitude"]].values, dtype=torch.float32, device=device)
        neg_pool = build_neg_pool(model, cfg, device)
        val_fn = make_val_fn(model, feats_mmap, val_idx, val_rows, grid, device)

        soft = cfg.get("soft_tau_km")
        soft_tau = float(soft) if soft is not None else None

        print(
            f"train {run_name} n={len(train_df)} neg={cfg.get('negatives')} "
            f"train_loc={cfg.get('train_loc', True)} soft_tau={soft_tau}",
            flush=True,
        )
        model, best_val = geoclip_train(
            model,
            f_tr,
            g_tr,
            neg_pool,
            train_loc=bool(cfg.get("train_loc", True)),
            soft_tau_km=soft_tau,
            epochs=int(cfg.get("epochs", 15)),
            batch_size=int(cfg.get("batch_size", 512)),
            n_neg=int(cfg.get("n_neg", 4096)),
            sigma_km=float(cfg.get("sigma_km", 1.0)),
            lr_mlp=float(cfg.get("lr_mlp", 1e-4)),
            lr_loc=float(cfg.get("lr_loc", 3e-5)),
            val_fn=val_fn,
            seed=int(cfg.get("seed", 0)),
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        log = {"run_name": run_name, "best_val_median_km_subset5k": best_val, "config": cfg}
        (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
        (RESULTS / "metrics" / f"{run_name}_train_log.json").write_text(json.dumps(log, indent=2))
        print(f"saved {ckpt_path} best_val_median={best_val:.3f} km", flush=True)

    model.eval()
    export_predictions(model, table, feats_mmap, grid, run_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    try:
        load_feature_paths(cfg.get("features", "clip_vitl14"))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    run_training(cfg, args.config, args.device)


if __name__ == "__main__":
    main()