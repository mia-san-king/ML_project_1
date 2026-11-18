"""Train and evaluate temporal towers from aligned, validated feature caches.

No import of lib.paths: importing this module never creates data directories.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import torch

from models.geoclip_time.lib.temporal_data import load_temporal_data
from models.geoclip_time.lib.temporal_factory import DEFAULT_ARCHITECTURE, build_temporal_model
from models.geoclip_time.lib.temporal_ops import predict_temporal, temporal_loss


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TIME = {"year_center": 1950.0, "year_scale": 100.0, "min_year": 1800, "max_year": 2026}
EVAL_SPLITS = ("val", "test_in", "test_city", "test_region")


def _configuration(cfg):
    out = {
        "run_name": "T1_dual_time", "features": "clip_vitl14", "epochs": 15,
        "batch_size": 32, "n_neg": 128, "val_limit": 256, "seed": 0,
        "lr_mlp": 1e-4, "lr_loc": 3e-5, "train_loc": True,
        "soft_tau_km": None, "sigma_km": 0.0,
        "negatives": "swiss_grid", "gallery": "geo/grid_1km.npy",
        "query_chunk": 16, "gallery_chunk": 256, "allow_partial_features": False,
        **deepcopy(cfg),
    }
    out["model"] = {**deepcopy(DEFAULT_ARCHITECTURE), **out.get("model", {})}
    out["time"] = {**DEFAULT_TIME, **out.get("time", {})}
    if not isinstance(out["run_name"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", out["run_name"]):
        raise ValueError("run_name must be a simple filename without directory separators")
    for key in ("epochs", "batch_size", "query_chunk", "gallery_chunk"):
        if isinstance(out[key], bool) or not isinstance(out[key], int) or out[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(out["n_neg"], int) or out["n_neg"] < 0:
        raise ValueError("n_neg must be a nonnegative integer")
    if out["val_limit"] is not None and (not isinstance(out["val_limit"], int) or out["val_limit"] < 1):
        raise ValueError("val_limit must be a positive integer or null")
    if out["sigma_km"] != 0:
        raise ValueError("this runner requires sigma_km=0 to retain exact positive GPS")
    if out["negatives"] != "swiss_grid":
        raise ValueError("this runner supports negatives=swiss_grid")
    if out["model"]["mode"] not in ("both", "image_only", "none", "direct"):
        raise ValueError("model.mode must be both, image_only, none, or direct")
    for key in ("lr_mlp", "lr_loc"):
        if not np.isfinite(out[key]) or out[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("allow_partial_features", "train_loc"):
        if not isinstance(out[key], bool):
            raise ValueError(f"{key} must be boolean")
    if out["soft_tau_km"] is not None and (not np.isfinite(out["soft_tau_km"]) or out["soft_tau_km"] <= 0):
        raise ValueError("soft_tau_km must be positive or null")
    return out


def _paths(cfg, data_root=None, split_csv=None):
    root_value = data_root or cfg.get("data_root") or os.environ.get("P2_DATA")
    if root_value is None:
        raise ValueError("set P2_DATA, data_root in configuration, or --data-root")
    root = Path(root_value).expanduser().resolve()
    split = Path(split_csv or cfg.get("split_csv") or PROJECT_ROOT / "data/splits/main.csv.gz").expanduser()
    return root, split.resolve()


def _gallery(root, cfg):
    path = Path(cfg["gallery"]).expanduser()
    if not path.is_absolute():
        path = root / path
    grid = np.load(path, allow_pickle=False)
    if grid.ndim != 2 or grid.shape[1] != 2 or len(grid) == 0:
        raise ValueError("gallery must have nonempty shape [N,2] in latitude,longitude order")
    if not np.isfinite(grid).all() or np.any(np.abs(grid[:, 0]) > 90) or np.any(np.abs(grid[:, 1]) > 180):
        raise ValueError("gallery contains invalid GPS coordinates")
    return np.asarray(grid, dtype=np.float32)


def _require_coverage(data, splits, allow_partial):
    for split in splits:
        coverage = data.audit["coverage"].get(split, {"expected": 0, "available": 0, "missing": 0})
        if coverage["missing"] and not allow_partial:
            raise ValueError(f"partial feature coverage for {split}: {coverage}; explicitly set allow_partial_features: true to continue")
        if not coverage["available"]:
            raise ValueError(f"no available rows for required split {split}")


def _load_training_data(cfg, root, split):
    data = load_temporal_data(root, split, feature_name=cfg["features"], time_config=cfg["time"])
    _require_coverage(data, ("train", "val"), cfg["allow_partial_features"])
    if data.features.shape[1] != cfg["model"]["feature_dim"]:
        raise ValueError("cached feature dimension does not match model configuration")
    return data


def check_training_data(cfg, *, data_root=None, split_csv=None):
    """Validate prerequisites without constructing a model or writing any files."""
    cfg = _configuration(cfg)
    root, split = _paths(cfg, data_root, split_csv)
    data = _load_training_data(cfg, root, split)
    grid = _gallery(root, cfg)
    audit = deepcopy(data.audit)
    audit["gallery_rows"] = int(len(grid))
    audit["validation_limit"] = cfg["val_limit"]
    return audit


def _assert_outputs_available(paths, overwrite):
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("outputs already exist (explicit overwrite required): " + ", ".join(existing))


def _json_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _distance_errors(rows, predictions):
    # Kept pure: lib.geo imports lib.paths and creates server directories.
    truth = np.radians(rows[["latitude", "longitude"]].to_numpy(dtype=float))
    pred = np.radians(np.asarray(predictions, dtype=float))
    delta = pred - truth
    h = np.sin(delta[:, 0] / 2) ** 2 + np.cos(truth[:, 0]) * np.cos(pred[:, 0]) * np.sin(delta[:, 1] / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _summary(errors):
    out = {"n": int(len(errors)), "median_km": float(np.median(errors)), "mean_km": float(np.mean(errors)), "p90_km": float(np.percentile(errors, 90))}
    out.update({f"acc@{threshold}km": float(np.mean(errors <= threshold) * 100) for threshold in (1, 5, 10, 25, 50, 200, 750, 2500)})
    return out


class _FeatureRows:
    """A lazy subset of a feature mmap, sliced by the retrieval query loop."""

    def __init__(self, features, indices):
        self.features, self.indices = features, indices
        self.shape = (len(indices), features.shape[1])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        return np.asarray(self.features[self.indices[key]], dtype=np.float32)


def _predict_rows(model, data, positions, grid, cfg):
    indices = data.table.iloc[positions].feat_idx.to_numpy(dtype=np.int64)
    # Preserve mmap-sized query batches while sharing one gallery base cache.
    features = _FeatureRows(data.features, indices)
    with torch.inference_mode():
        pred, _, _, _ = predict_temporal(
            model, features, data.times[positions], grid,
            top_k=1, query_chunk=cfg["query_chunk"], gallery_chunk=cfg["gallery_chunk"],
        )
    return pred


def load_temporal_checkpoint(path, *, device="cpu", model_factory=build_temporal_model):
    """Reconstruct architecture and strict state offline, without initial weights."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported temporal checkpoint format")
    for key in ("architecture", "time_config", "model_state", "config"):
        if key not in payload:
            raise ValueError(f"checkpoint missing {key}")
    model = model_factory(payload["architecture"], pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), payload


def run_training(cfg, *, data_root=None, split_csv=None, device="cpu", overwrite=False, model_factory=build_temporal_model):
    """Train on train only; select and save the best exact-val checkpoint."""
    cfg = _configuration(cfg)
    root, split = _paths(cfg, data_root, split_csv)
    cfg["data_root"], cfg["split_csv"] = str(root), str(split)
    checkpoint_path = root / "ckpt" / f"{cfg['run_name']}.pt"
    log_path = root / "results/metrics" / f"{cfg['run_name']}_train_log.json"
    _assert_outputs_available((checkpoint_path, log_path), overwrite)
    data = _load_training_data(cfg, root, split)
    grid = _gallery(root, cfg)
    train = np.flatnonzero(data.table.split.to_numpy() == "train")
    validation = np.flatnonzero(data.table.split.to_numpy() == "val")
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    if cfg["val_limit"] is not None and len(validation) > cfg["val_limit"]:
        validation = np.sort(rng.choice(validation, size=cfg["val_limit"], replace=False))
    validation_ids = data.table.iloc[validation].id.tolist()
    torch.manual_seed(seed)
    model = model_factory(cfg["model"], pretrained=True).to(device)
    if not cfg["train_loc"]:
        model.location_encoder.requires_grad_(False)
    location_ids = {id(p) for p in model.location_encoder.parameters()}
    groups = []
    for is_location, rate in ((False, cfg["lr_mlp"]), (True, cfg["lr_loc"])):
        parameters = [p for p in model.parameters() if p.requires_grad and (id(p) in location_ids) == is_location]
        if parameters:
            groups.append({"params": parameters, "lr": rate})
    optimizer = torch.optim.Adam(groups)
    history, best_score, best_state, best_epoch = [], math.inf, None, None
    print(f"train {cfg['run_name']}: train={len(train)}, val={len(validation)} (limit={cfg['val_limit']}), mode={cfg['model']['mode']}", flush=True)
    print("coverage=" + json.dumps(data.audit["coverage"], sort_keys=True), flush=True)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = rng.permutation(train)
        total_loss, seen, steps, last_size = 0.0, 0, 0, 0
        for start in range(0, len(order), cfg["batch_size"]):
            positions = order[start:start + cfg["batch_size"]]
            rows = data.table.iloc[positions]
            features = torch.as_tensor(np.asarray(data.features[rows.feat_idx.to_numpy(dtype=np.int64)], dtype=np.float32), device=device)
            gps = torch.as_tensor(rows[["latitude", "longitude"]].to_numpy(dtype=np.float32), device=device)
            times = torch.as_tensor(np.asarray(data.times[positions], dtype=np.float32), device=device)
            negative_idx = rng.integers(0, len(grid), size=cfg["n_neg"])
            negatives = torch.as_tensor(grid[negative_idx], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = temporal_loss(model, features, gps, times, negatives, soft_tau_km=cfg["soft_tau_km"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at epoch {epoch}, batch {steps}")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100))
            last_size = len(positions)
            total_loss += float(loss.detach()) * last_size
            seen += last_size
            steps += 1
            if epoch == 1 and steps in (1, 100):
                print(f"epoch=1 step={steps} seen={seen} mean_train_loss={total_loss / seen:.6f}", flush=True)
        model.eval()
        predicted = _predict_rows(model, data, validation, grid, cfg)
        score = float(np.median(_distance_errors(data.table.iloc[validation], predicted)))
        if not math.isfinite(score):
            raise FloatingPointError("nonfinite validation metric")
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "train_examples": seen,
                        "optimizer_steps": steps, "last_batch_size": last_size, "val_median_km": score})
        print(f"epoch={epoch} loss={total_loss / seen:.6f} val_median_km={score:.6f}", flush=True)
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    model.eval()
    payload = {"format_version": 1, "architecture": cfg["model"], "time_config": cfg["time"],
               "model_state": best_state, "config": cfg, "history": history, "epoch": best_epoch,
               "validation_ids": validation_ids, "best_val_median_km": best_score, "audit": data.audit}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    log = {key: value for key, value in payload.items() if key != "model_state"}
    log["run_name"] = cfg["run_name"]
    _json_write(log_path, log)
    return {"checkpoint": checkpoint_path, "log": log_path, "model": model,
            "best_val_median_km": best_score, "validation_ids": validation_ids}


def run_prediction(checkpoint, *, data_root=None, split_csv=None, splits=EVAL_SPLITS,
                   device="cpu", overwrite=False, allow_partial_features=False,
                   model_factory=build_temporal_model):
    """Export selected evaluation splits using checkpoint-bound preprocessing."""
    # Read metadata before model construction so collisions fail cheaply.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = _configuration(payload["config"])
    cfg["time"] = payload["time_config"]
    cfg["model"] = payload["architecture"]
    root, split = _paths(cfg, data_root, split_csv)
    pred_path = root / "results/preds" / f"{cfg['run_name']}.parquet"
    metrics_path = root / "results/metrics" / f"{cfg['run_name']}_summary.json"
    _assert_outputs_available((pred_path, metrics_path), overwrite)
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or not set(splits).issubset(EVAL_SPLITS):
        raise ValueError("splits must be unique names from val, test_in, test_city, test_region")
    data = load_temporal_data(root, split, feature_name=cfg["features"], time_config=cfg["time"])
    _require_coverage(data, splits, allow_partial_features)
    grid = _gallery(root, cfg)
    positions = np.flatnonzero(data.table.split.isin(splits).to_numpy())
    model, _ = load_temporal_checkpoint(checkpoint, device=device, model_factory=model_factory)
    predicted = _predict_rows(model, data, positions, grid, cfg)
    rows = data.table.iloc[positions]
    out = rows[["id", "split"]].copy()
    out["lat_pred"], out["lon_pred"] = predicted[:, 0], predicted[:, 1]
    errors = _distance_errors(rows, predicted)
    metrics = []
    for name in splits:
        metrics.append({"split": name, "view_type": "all", **_summary(errors[rows.split.to_numpy() == name])})
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pred_path.with_name(pred_path.name + ".tmp")
    out.to_parquet(temporary, index=False)
    temporary.replace(pred_path)
    validation_summary = next((row for row in metrics if row["split"] == "val"), {})
    _json_write(metrics_path, {"splits": metrics, "val": validation_summary, "audit": data.audit,
                              "checkpoint": str(Path(checkpoint).resolve()), "time_config": cfg["time"]})
    return {"predictions": pred_path, "metrics": metrics_path}
