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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.baseline_infer import ALL_EVAL_SPLITS, load_feature_tables, save_json_metrics
from lib.geo import haversine_km
from lib.geoclip_ops import TimeBranch, predict_feats
from lib.geoclip_ops import train as geoclip_train
from lib.paths import CKPT, GEO, RESULTS
from lib.time_feats import ensure_time_feats, load_time_feats_for_table

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


def load_init_checkpoint(model, cfg, device: str):
    init = cfg.get("init_ckpt")
    if init:
        path = CKPT / f"{init}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        print(f"init from {path}", flush=True)


def load_time_branch(run_name: str, device: str) -> TimeBranch | None:
    path = CKPT / f"{run_name}_time.pt"
    if not path.is_file():
        return None
    branch = TimeBranch().to(device)
    branch.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return branch


def make_val_fn(
    feats_mmap,
    val_idx: np.ndarray,
    val_truth: pd.DataFrame,
    val_time: np.ndarray,
    grid,
    device: str,
    time_branch: TimeBranch | None,
):
    grid = np.asarray(grid, dtype=float)
    feats = np.asarray(feats_mmap[val_idx], dtype=np.float32)
    lat = val_truth.latitude.values
    lon = val_truth.longitude.values
    tfeat = np.asarray(val_time, dtype=np.float32) if time_branch is not None else None

    def val_fn(model, branch: TimeBranch | None = None):
        m = model
        b = branch if branch is not None else time_branch
        m.eval()
        with torch.no_grad():
            preds, _, _, _ = predict_feats(
                m, feats, grid, top_k=1, chunk=256, time_branch=b, time_feats=tfeat
            )
        err = haversine_km(lat, lon, preds[:, 0], preds[:, 1])
        return float(np.median(err))

    return val_fn


@torch.no_grad()
def export_predictions(
    model,
    table,
    feats_mmap,
    grid,
    run_name: str,
    time_branch: TimeBranch | None = None,
):
    pred_path = RESULTS / "preds" / f"{run_name}.parquet"
    if pred_path.is_file():
        print(f"skip export {run_name} (exists)", flush=True)
        return
    sub = table[table.split.isin(ALL_EVAL_SPLITS)].copy()
    order = sub.feat_idx.values
    f = np.asarray(feats_mmap[order], dtype=np.float32)
    tfeat = load_time_feats_for_table(sub, feats_mmap) if time_branch is not None else None
    preds, _, _, _ = predict_feats(
        model,
        f,
        np.asarray(grid, dtype=float),
        top_k=1,
        chunk=1024,
        time_branch=time_branch,
        time_feats=tfeat,
    )
    out = sub[["id", "split"]].copy()
    out["lat_pred"] = preds[:, 0]
    out["lon_pred"] = preds[:, 1]
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pred_path, index=False)
    truth = table[["id", "latitude", "longitude", "view_type", "block"]]
    from lib.metrics import evaluate

    metrics, _ = evaluate(out, truth)
    save_json_metrics(run_name, metrics)
    print(f"wrote {pred_path}", flush=True)


def run_training(cfg: dict, config_path: Path, device: str, force: bool = False):
    run_name = cfg.get("run_name", config_path.stem)
    feat_name = cfg.get("features", "clip_vitl14")
    feat_path, _ = load_feature_paths(feat_name)
    feats_mmap = np.load(feat_path, mmap_mode="r")

    _, _, table = load_feature_tables()
    train_df = table[table.split == "train"]
    if len(train_df) == 0:
        raise RuntimeError("no train rows")

    use_time = bool(cfg.get("time", False))
    if use_time:
        ensure_time_feats()

    grid = np.load(GEO / "grid_1km.npy")
    val_idx = val_subset_indices(table, int(cfg.get("seed", 0)))
    val_rows = table[table.feat_idx.isin(val_idx)].set_index("feat_idx").loc[val_idx].reset_index()
    val_time = load_time_feats_for_table(val_rows, feats_mmap) if use_time else None

    ckpt_path = CKPT / f"{run_name}.pt"
    time_ckpt_path = CKPT / f"{run_name}_time.pt"
    model = GeoCLIP().to(device)
    time_branch = TimeBranch().to(device) if use_time else None

    if use_time:
        need_train = force or not time_ckpt_path.is_file()
        if ckpt_path.is_file():
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        else:
            load_init_checkpoint(model, cfg, device)
        if time_ckpt_path.is_file() and not force:
            time_branch.load_state_dict(torch.load(time_ckpt_path, map_location=device, weights_only=True))
            need_train = False
    else:
        need_train = force or not ckpt_path.is_file()
        if ckpt_path.is_file() and not force:
            print(f"load checkpoint {ckpt_path}", flush=True)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            need_train = False

    if need_train:
        f_tr = torch.as_tensor(np.asarray(feats_mmap[train_df.feat_idx.values], dtype=np.float32), device=device)
        g_tr = torch.as_tensor(train_df[["latitude", "longitude"]].values, dtype=torch.float32, device=device)
        t_tr = None
        if use_time:
            t_tr = torch.as_tensor(load_time_feats_for_table(train_df, feats_mmap), device=device)
        neg_pool = build_neg_pool(model, cfg, device)
        val_fn = make_val_fn(feats_mmap, val_idx, val_rows, val_time, grid, device, time_branch)

        soft = cfg.get("soft_tau_km")
        soft_tau = float(soft) if soft is not None else None
        train_loc = bool(cfg.get("train_loc", True))
        if cfg.get("freeze_loc") is True:
            train_loc = False

        print(
            f"train {run_name} n={len(train_df)} time={use_time} neg={cfg.get('negatives')} "
            f"train_loc={train_loc} soft_tau={soft_tau}",
            flush=True,
        )
        model, best_val = geoclip_train(
            model,
            f_tr,
            g_tr,
            neg_pool,
            train_loc=train_loc,
            soft_tau_km=soft_tau,
            epochs=int(cfg.get("epochs", 15)),
            batch_size=int(cfg.get("batch_size", 512)),
            n_neg=int(cfg.get("n_neg", 4096)),
            sigma_km=float(cfg.get("sigma_km", 1.0)),
            lr_mlp=float(cfg.get("lr_mlp", 1e-4)),
            lr_loc=float(cfg.get("lr_loc", 3e-5)),
            val_fn=val_fn,
            seed=int(cfg.get("seed", 0)),
            time_branch=time_branch,
            t_tr=t_tr,
            time_dropout=float(cfg.get("time_dropout", 0.0)),
            lr_time=float(cfg.get("lr_time", 1e-3)),
            train_mlp=bool(cfg.get("train_mlp", True)),
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        if time_branch is not None:
            torch.save(time_branch.state_dict(), time_ckpt_path)
        log = {"run_name": run_name, "best_val_median_km_subset5k": best_val, "config": cfg}
        (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
        (RESULTS / "metrics" / f"{run_name}_train_log.json").write_text(json.dumps(log, indent=2))
        print(f"saved {ckpt_path} best_val_median={best_val:.3f} km", flush=True)
    else:
        print(f"load existing {run_name} checkpoints", flush=True)

    model.eval()
    if time_branch is not None:
        time_branch.eval()
    export_predictions(model, table, feats_mmap, grid, run_name, time_branch=time_branch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    try:
        load_feature_paths(cfg.get("features", "clip_vitl14"))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    run_training(cfg, args.config, args.device, force=args.force)


if __name__ == "__main__":
    main()