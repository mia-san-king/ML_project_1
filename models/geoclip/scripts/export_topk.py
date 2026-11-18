#!/usr/bin/env python3
"""Export top-k logits for a fine-tuned run (reliability features)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from geoclip import GeoCLIP

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib.feature_tables import load_feature_tables
from models.geoclip.lib.geoclip_ops import predict_feats
from lib.paths import CKPT, FEATS, GEO, RESULTS

EVAL_SPLITS = {"val", "test_in", "test_city", "test_region"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="A4_tau5")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = RESULTS / "preds" / f"{args.run_name}_topk.npz"
    if out.is_file() and not args.force:
        print(f"skip {out}", flush=True)
        return

    ckpt = CKPT / f"{args.run_name}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    feats_mmap = np.load(FEATS / "clip_vitl14.npy", mmap_mode="r")
    _, _, table = load_feature_tables()
    sub = table[table.split.isin(EVAL_SPLITS)].copy()
    f = np.asarray(feats_mmap[sub.feat_idx.values], dtype=np.float32)
    gallery = np.load(GEO / "grid_1km.npy")

    model = GeoCLIP().to(args.device)
    model.load_state_dict(torch.load(ckpt, map_location=args.device, weights_only=True))
    model.eval()

    print(f"export top-{args.top_k} n={len(sub)} run={args.run_name}", flush=True)
    _, top_idx, top_logit, lse = predict_feats(model, f, gallery, top_k=args.top_k, chunk=512)

    np.savez_compressed(
        out,
        ids=sub.id.values.astype(np.int64),
        split=sub.split.values,
        top_idx=top_idx.astype(np.int32),
        top_logit=top_logit.astype(np.float32),
        logsumexp=lse.astype(np.float32),
        gallery="grid_1km",
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()