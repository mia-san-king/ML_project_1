#!/usr/bin/env python3
"""Run A1 (B1 + regional prior) with best hyperparameters from search_a1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from geoclip import GeoCLIP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.baseline_infer import ensure_mlp512, load_feature_tables, run_geoclip_baseline
from lib.geoclip_ops import regional_penalty
from lib.paths import FEATS, RESULTS

RADIUS_KM = 179.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--scale-km", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    search_path = RESULTS / "metrics" / "a1_search.json"
    if not search_path.is_file():
        raise FileNotFoundError(f"run search_a1.py first: {search_path}")
    search = json.loads(search_path.read_text())
    best = search["best"]
    lam = args.lam if args.lam is not None else best["lam"]
    scale_km = args.scale_km if args.scale_km is not None else best["scale_km"]

    feats, ids, table = load_feature_tables()
    model = GeoCLIP().to(args.device).eval()
    mlp512 = ensure_mlp512(model, feats, ids)
    gallery = np.load(FEATS / "gallery_mp16_gps.npy")
    penalty = regional_penalty(gallery, radius_km=RADIUS_KM, scale_km=scale_km)

    run_geoclip_baseline(
        model,
        table,
        mlp512,
        gallery,
        "mp16",
        "A1",
        args.device,
        penalty=penalty,
        lam=lam,
        skip_if_exists=not args.force,
    )


if __name__ == "__main__":
    main()