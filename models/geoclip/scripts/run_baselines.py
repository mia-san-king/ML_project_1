#!/usr/bin/env python3
"""B1/B2/B3 baselines on cached CLIP features (all non-nadir splits; skip existing outputs)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from geoclip import GeoCLIP

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.geoclip.lib.baseline_infer import ensure_gallery_emb, ensure_mlp512, run_geoclip_baseline
from models.baselines.scripts.run_knn import run_b2
from lib.feature_tables import load_feature_tables
from lib.paths import GEO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=("b1", "b2", "b3", "all"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    feats, ids, table = load_feature_tables()
    which = {args.which} if args.which != "all" else {"b1", "b2", "b3"}

    model = None
    if which & {"b1", "b3"}:
        model = GeoCLIP().to(args.device).eval()
        mlp512 = ensure_mlp512(model, feats, ids)
    else:
        mlp512 = None

    if "b1" in which:
        gallery = model.gps_gallery.detach().cpu().numpy()
        ensure_gallery_emb(model, gallery, "mp16")
        run_geoclip_baseline(model, table, mlp512, gallery, "mp16", "B1", args.device)

    if "b3" in which:
        grid = np.load(GEO / "grid_1km.npy")
        run_geoclip_baseline(model, table, mlp512, grid, "grid1km", "B3", args.device)

    if "b2" in which:
        run_b2(table, feats, args.device)


if __name__ == "__main__":
    main()