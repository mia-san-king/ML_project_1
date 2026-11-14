#!/usr/bin/env python3
"""GeoCLIP fine-tuning entry point (requires cached CLIP features)."""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.paths import FEATS


def load_feature_paths(name="clip_vitl14"):
    feat_path = FEATS / f"{name}.npy"
    ids_path = FEATS / f"{name}_ids.npy"
    if not feat_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"missing {feat_path} or {ids_path}; run S6 feature caching after images are downloaded"
        )
    return feat_path, ids_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    name = cfg.get("run_name", args.config.stem)

    try:
        feat_path, ids_path = load_feature_paths(cfg.get("features", "clip_vitl14"))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    print(f"run={name} config={args.config} features={feat_path} ids={ids_path}")
    print("S6 features present; full GPU training loop to be wired after priority-1 images are cached.")
    sys.exit(0)


if __name__ == "__main__":
    main()