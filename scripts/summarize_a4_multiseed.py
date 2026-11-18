#!/usr/bin/env python3
"""Aggregate val + test splits for A4_tau5 seeds 0,1,2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.paths import RESULTS

RUNS = ["A4_tau5", "A4_tau5_s1", "A4_tau5_s2"]
SPLITS = ["val", "test_in", "test_city", "test_region"]
KEYS = ("median_km", "mean_km", "acc@25km")


def main():
    rows = []
    for run in RUNS:
        path = RESULTS / "metrics" / f"{run}_summary.json"
        if not path.is_file():
            print(f"missing {path}", file=sys.stderr)
            continue
        data = json.loads(path.read_text())
        for split in SPLITS:
            match = [r for r in data["splits"] if r["split"] == split and r["view_type"] == "all"]
            if not match:
                continue
            r = match[0]
            seed = json.loads((RESULTS / "metrics" / f"{run}_train_log.json").read_text()).get("config", {}).get("seed")
            rows.append(
                {
                    "run": run,
                    "seed": seed,
                    "split": split,
                    **{k: r[k] for k in KEYS if k in r},
                }
            )

    out = RESULTS / "metrics" / "A4_tau5_multiseed.json"
    # means per split across available runs
    summary = {}
    import statistics

    for split in SPLITS:
        vals = [x for x in rows if x["split"] == split]
        if not vals:
            continue
        summary[split] = {
            "n_seeds": len(vals),
            "median_km_mean": statistics.mean(x["median_km"] for x in vals),
            "median_km_stdev": statistics.stdev(x["median_km"] for x in vals) if len(vals) > 1 else 0.0,
            "acc@25km_mean": statistics.mean(x["acc@25km"] for x in vals),
        }

    payload = {"per_run": rows, "aggregate": summary}
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    for split in SPLITS:
        if split in summary:
            s = summary[split]
            print(
                f"{split}: median {s['median_km_mean']:.2f}±{s['median_km_stdev']:.2f} km, "
                f"acc25 {s['acc@25km_mean']:.1f}%"
            )


if __name__ == "__main__":
    main()