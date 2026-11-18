#!/usr/bin/env python3
"""Train time-conditioned GeoCLIP from precomputed CLIP feature caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, help="data root (otherwise configuration or P2_DATA)")
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--device", default=None, help="cpu, cuda, or mps; defaults to CUDA when available")
    parser.add_argument("--check-data", action="store_true", help="validate input coverage and grid; write nothing and load no model")
    parser.add_argument("--overwrite-run", action="store_true", help="replace this run's checkpoint and training log only")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch
    import yaml
    from models.geoclip_time.lib.temporal_training import check_training_data, run_training

    cfg = yaml.safe_load(args.config.read_text())
    if not isinstance(cfg, dict):
        parser.error("configuration must be a YAML mapping")
    cfg.setdefault("run_name", args.config.stem)
    try:
        if args.check_data:
            audit = check_training_data(cfg, data_root=args.data_root, split_csv=args.split_csv)
            print(json.dumps(audit, indent=2))
        else:
            result = run_training(cfg, data_root=args.data_root, split_csv=args.split_csv,
                                  device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                                  overwrite=args.overwrite_run)
            print(f"saved {result['checkpoint']} and {result['log']}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
