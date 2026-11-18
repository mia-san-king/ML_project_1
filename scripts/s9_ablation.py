#!/usr/bin/env python3
"""Summarize S9 val ablation from B1/A1/B3/B2 metric JSON files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.paths import RESULTS


def pick_val(payload: dict) -> dict:
    val = payload.get("val", {})
    return {k: val[k] for k in ("n", "median_km", "mean_km", "p90_km", "acc@25km") if k in val}


def main():
    rows = []
    for name, label in [("B1", "B1 global"), ("A1", "A1 + prior"), ("B3", "A2 grid"), ("B2", "B2 kNN")]:
        path = RESULTS / "metrics" / f"{name}_summary.json"
        if not path.is_file():
            continue
        rows.append({"id": name, "label": label, **pick_val(json.loads(path.read_text()))})
    out = RESULTS / "metrics" / "s9_val_ablation.json"
    search = RESULTS / "metrics" / "a1_search.json"
    payload = {"val_ablation": rows}
    if search.is_file():
        payload["a1_best"] = json.loads(search.read_text()).get("best")
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()