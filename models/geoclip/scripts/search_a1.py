#!/usr/bin/env python3
"""Grid-search regional prior (A1) on val; uses cached mlp512 + mp16 gallery."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from geoclip import GeoCLIP

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib.feature_tables import load_feature_tables
from models.geoclip.lib.baseline_infer import ensure_mlp512, predict_splits, val_metrics_from_preds
from models.geoclip.lib.geoclip_ops import regional_penalty
from lib.paths import FEATS, RESULTS

LAMS = (1.0, 3.0, 10.0, 30.0)
SCALES_KM = (25.0, 50.0, 100.0)
RADIUS_KM = 179.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = RESULTS / "metrics" / "a1_search.json"
    if out_path.is_file() and not args.force:
        print(f"skip search ({out_path} exists)", flush=True)
        return

    feats, ids, table = load_feature_tables()
    model = GeoCLIP().to(args.device).eval()
    mlp512 = ensure_mlp512(model, feats, ids)
    gps_path = FEATS / "gallery_mp16_gps.npy"
    if not gps_path.is_file():
        raise FileNotFoundError(f"run B1 first: missing {gps_path}")
    gallery = np.load(gps_path)

    val_table = table[table.split == "val"]
    grid_results = []
    best = None
    best_median = float("inf")

    for scale_km in SCALES_KM:
        penalty = regional_penalty(gallery, radius_km=RADIUS_KM, scale_km=scale_km)
        for lam in LAMS:
            sub, preds, _, _, _, _ = predict_splits(
                model,
                val_table,
                mlp512,
                "mp16",
                split_names={"val"},
                device=args.device,
                penalty=penalty,
                lam=lam,
                top_k=1,
            )
            metrics = val_metrics_from_preds(sub, preds)
            row = {
                "lam": lam,
                "scale_km": scale_km,
                "radius_km": RADIUS_KM,
                "val": metrics,
            }
            grid_results.append(row)
            print(f"lam={lam} scale_km={scale_km} median={metrics['median_km']:.2f} acc25={metrics['acc@25km']:.1f}", flush=True)
            if metrics["median_km"] < best_median:
                best_median = metrics["median_km"]
                best = row

    payload = {"grid": grid_results, "best": best}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {out_path} best lam={best['lam']} scale_km={best['scale_km']}", flush=True)


if __name__ == "__main__":
    main()