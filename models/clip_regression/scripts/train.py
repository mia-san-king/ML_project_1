"""Fit T4 on frozen CLIP features and date intervals, then export Swiss coordinates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
import torch

ROOT = Path(__file__).resolve().parents[3]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from models.clip_regression.lib.regression import LinearTimeRegressor, clip_to_switzerland
from models.geoclip_time.lib.temporal_data import load_temporal_data
from models.geoclip_time.lib.temporal_training import EVAL_SPLITS, _distance_errors, _summary

LV95_ORIGIN_M = np.array([2_600_000., 1_200_000.])
CHUNK_SIZE = 4096


def fit_linear(clip, dates, train_rows, target_km):
    """Fit the affine layer by pivoted QR on the training design matrix."""
    if not len(train_rows) or target_km.shape != (len(train_rows), 2):
        raise ValueError("T4 requires two coordinate targets per training image")
    width = clip.shape[1]
    design = torch.empty((len(train_rows), width + 3 + 1), dtype=torch.float64)
    for start in range(0, len(train_rows), CHUNK_SIZE):
        rows = train_rows[start:start + CHUNK_SIZE]
        batch = design[start:start + len(rows)]
        batch[:, :width] = torch.from_numpy(clip[rows])
        batch[:, width:-1] = torch.from_numpy(dates[rows])
        batch[:, -1] = 1.

    targets = torch.as_tensor(target_km, dtype=torch.float64)
    fit = torch.linalg.lstsq(design, targets, driver="gelsy")
    model = LinearTimeRegressor(width).double()
    with torch.no_grad():
        model.linear.weight.copy_(fit.solution[:-1].T)
        model.linear.bias.copy_(fit.solution[-1])
    return model.eval(), int(fit.rank)


def fit_and_predict(data_root, split_csv, output):
    if output.exists():
        raise FileExistsError(f"Choose a new results directory: {output}")
    data = load_temporal_data(data_root, split_csv)
    border = gpd.read_file(data_root / "geo/ne_admin0.gpkg").to_crs(2056).geometry.union_all()
    to_lv95 = Transformer.from_crs(4326, 2056, always_xy=True)
    to_wgs = Transformer.from_crs(2056, 4326, always_xy=True)
    rows = data.table
    train_rows = np.flatnonzero(rows.split.to_numpy() == "train")
    train = rows.iloc[train_rows]
    east, north = to_lv95.transform(train.longitude.to_numpy(), train.latitude.to_numpy())
    target_km = (np.column_stack((east, north)) - LV95_ORIGIN_M) / 1000.
    print(f"Fitting T4 on {len(train):,} training images", flush=True)
    model, rank = fit_linear(data.features, data.times, train_rows, target_km)

    frames = []
    metrics = {}
    for split in EVAL_SPLITS:
        positions = np.flatnonzero(rows.split.to_numpy() == split)
        if not len(positions):
            raise ValueError(f"No cached evaluation images for {split}")
        pred = rows.iloc[positions][["id", "split", "latitude", "longitude"]].copy()
        offsets = []
        with torch.inference_mode():
            for start in range(0, len(positions), CHUNK_SIZE):
                batch = positions[start:start + CHUNK_SIZE]
                image = torch.from_numpy(data.features[batch]).double()
                date = torch.from_numpy(data.times[batch]).double()
                offsets.append(model(image, date).numpy())
        raw_xy = np.concatenate(offsets) * 1000. + LV95_ORIGIN_M
        xy = clip_to_switzerland(raw_xy, border)
        lon, lat = to_wgs.transform(xy[:, 0], xy[:, 1])
        pred["lat_pred"], pred["lon_pred"] = lat, lon
        pred["has_date"] = data.times[positions, 2].astype(bool)
        pred["was_projected"] = (raw_xy != xy).any(axis=1)
        pred["error_km"] = _distance_errors(pred, np.column_stack((lat, lon)))
        metrics[split] = {**_summary(pred.error_km), "projected": int(pred.was_projected.sum())}
        frames.append(pred)
        print(f"{split}: median {metrics[split]['median_km']:.3f} km", flush=True)

    date_config = data.audit["dates"]["config"]
    info = {
        "model": "T4", "solver": "ordinary least squares (pivoted QR, float64)",
        "train_n": len(train), "train_id_sha256": hashlib.sha256(
            np.sort(train.id.to_numpy(dtype="<i8")).tobytes()).hexdigest(),
        "design_rank": rank, "parameters": sum(p.numel() for p in model.parameters()),
        "coverage": data.audit["coverage"], "time_config": date_config, "splits": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        dest = Path(staging)
        torch.save({
            "model": "T4", "feature_name": "clip_vitl14", "feature_dim": data.features.shape[1],
            "model_state": model.state_dict(), "time_config": date_config,
            "output_crs": "EPSG:2056", "origin_m": LV95_ORIGIN_M.tolist(), "scale_m": 1000.,
        }, dest / "T4.pt")
        pd.concat(frames, ignore_index=True).to_parquet(dest / "predictions.parquet", index=False)
        (dest / "summary.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
        dest.rename(output)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=ROOT / "data/splits/main.csv.gz")
    parser.add_argument("--output", type=Path, required=True, help="new directory for this run")
    args = parser.parse_args()
    fit_and_predict(args.data_root, args.split_csv, args.output)


if __name__ == "__main__":
    main()
