"""Refit the T0 and T1 parts of T5 outside a geographically withheld region."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import yaml

from models.geoclip_time.lib.temporal_training import run_training
from models.geoclip_time.scripts.rerank_references import _sha256


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "report/experiments/2026-11-18-geoclip-time/run"
CONFIGS = {"T0": "T0_no_time_full.yaml", "T1": "T1_dual_time_full.yaml"}


def verify_split(split, augmented_root):
    assigned = pd.read_csv(split, usecols=["id", "split"]).set_index("id", verify_integrity=True)
    ids = np.load(augmented_root / "features/clip_vitl14_ids.npy", allow_pickle=False)
    selected = assigned.loc[ids, "split"]
    unknown = set(selected) - {"train", "val", "query", "ignore"}
    if unknown or not {"train", "val", "query"}.issubset(set(selected)):
        raise ValueError(f"Unexpected or incomplete fold roles: {set(selected)}")
    for role in ("query", "val"):
        if (selected.index[selected == role] <= 0).any():
            raise ValueError(f"{role} IDs must be positive originals")
    train_ids = selected.index[selected == "train"].to_numpy()
    if (train_ids == 0).any() or not np.isin(-train_ids[train_ids < 0], train_ids[train_ids > 0]).all():
        raise ValueError("Every included variant must have an included training parent")


def fit_towers(augmented_root, assigned_split_csv, output_dir, *, device="cuda"):
    augmented_root, split, output = map(Path, (augmented_root, assigned_split_csv, output_dir))
    verify_split(split, augmented_root)
    output.mkdir(parents=True, exist_ok=True)
    files = {name: augmented_root / name for name in ("geo/grid_1km.npy", "meta/smapshot.parquet",
             "features/clip_vitl14.npy", "features/clip_vitl14_ids.npy")}
    files["assigned_split"] = split
    signature = {name: _sha256(path) for name, path in files.items()}
    paths = {}
    for name, config in CONFIGS.items():
        cfg_path = CONFIG_DIR / config
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg.update(run_name=name, data_root=str(augmented_root.resolve()), split_csv=str(split.resolve()))
        target = output / name
        ckpt = target / "ckpt" / f"{name}.pt"
        receipt = {**signature, "training_config": _sha256(cfg_path), "device": device}
        if target.exists():
            saved = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
            if saved["inputs"] != receipt or _sha256(ckpt) != saved["checkpoint_sha256"]:
                raise ValueError(f"Cannot reuse {name}: checkpoint or fold inputs changed")
        else:
            with TemporaryDirectory(dir=output) as temp:
                stage = Path(temp) / name
                result = run_training(cfg, output_root=stage, device=device)
                audit = {"inputs": receipt, "checkpoint_sha256": _sha256(result["checkpoint"]),
                         "best_val_median_km": result["best_val_median_km"]}
                (stage / "provenance.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
                del result
                stage.rename(target)
        paths[name] = ckpt
        print(f"READY {name}: {ckpt}", flush=True)
    return paths


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("augmented-root", "assigned-split-csv", "output-dir"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    torch.set_num_threads(4)
    fit_towers(args.augmented_root, args.assigned_split_csv, args.output_dir / "towers", device=args.device)
