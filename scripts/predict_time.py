#!/usr/bin/env python3
"""Predict evaluation splits with the saved model and its date preprocessing."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("val", "test_in", "test_city", "test_region"),
                        default=["val", "test_in", "test_city", "test_region"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-partial-features", action="store_true", help="explicitly export available rows when requested split features are incomplete")
    parser.add_argument("--overwrite-output", action="store_true", help="replace this run's prediction parquet and summary only")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from lib.temporal_training import run_prediction

    try:
        result = run_prediction(args.checkpoint, data_root=args.data_root, split_csv=args.split_csv,
                                splits=args.splits, device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                                overwrite=args.overwrite_output, allow_partial_features=args.allow_partial_features)
        print(f"wrote {result['predictions']} and {result['metrics']}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
