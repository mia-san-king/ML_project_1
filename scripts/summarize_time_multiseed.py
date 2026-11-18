#!/usr/bin/env python3
"""Aggregate multi-seed metrics for a time-aware run (T*)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import RESULTS

SPLITS = ("val", "test_in", "test_city", "test_region")


def load_split_metrics(run: str, split: str) -> dict:
    data = json.loads((RESULTS / "metrics" / f"{run}_summary.json").read_text())
    rows = [r for r in data["splits"] if r["split"] == split and r["view_type"] == "all"]
    return rows[0] if rows else {}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="T2")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    runs = [args.base if s == 0 else f"{args.base}_s{s}" for s in seeds]

    per_run = []
    for run, seed in zip(runs, seeds):
        for split in SPLITS:
            row = load_split_metrics(run, split)
            if not row:
                continue
            per_run.append(
                {
                    "run": run,
                    "seed": seed,
                    "split": split,
                    "median_km": row["median_km"],
                    "acc@25km": row["acc@25km"],
                }
            )
    df = pd.DataFrame(per_run)
    agg = {}
    for split in SPLITS:
        sub = df[df.split == split]
        if sub.empty:
            continue
        agg[split] = {
            "n_seeds": int(len(sub)),
            "median_km_mean": float(sub.median_km.mean()),
            "median_km_stdev": float(sub.median_km.std(ddof=0)),
            "acc@25km_mean": float(sub["acc@25km"].mean()),
        }
    out = {"per_run": per_run, "aggregate": agg, "base": args.base}
    path = RESULTS / "metrics" / f"{args.base}_multiseed.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}", flush=True)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()