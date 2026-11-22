"""Run the four predefined regional expert folds and compare the selectors."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from models.t6_router.region_splits import write_region_splits
from scripts.confidence.evaluate import sha256


def run(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    workspace = args.workspace.resolve()
    augmented = args.augmented_root.resolve()
    canonical = workspace / "data/splits/main.csv.gz"
    variants = augmented / "features/clip_vitl14_ids.npy"
    regions = ("dev1", "dev2", "dev3", "final")
    split_dir = root / "splits"
    if split_dir.exists():
        with TemporaryDirectory(dir=root) as temp:
            expected = write_region_splits(canonical_split=canonical, augmented_ids=variants,
                                           output_dir=Path(temp) / "splits")
            splits = {name: split_dir / path.name for name, path in expected.items()}
            for name, path in expected.items():
                if sha256(splits[name]) != sha256(path):
                    raise ValueError(f"Stored and regenerated assignments differ in {name}")
    else:
        splits = write_region_splits(canonical_split=canonical, augmented_ids=variants, output_dir=split_dir)
    if set(splits) != set(regions):
        raise ValueError("Expected three named development regions and one reserved region")
    if args.stage == "towers":
        from models.t6_router.region_towers import fit_towers
        for region in regions:
            print(f"STAGE towers {region}", flush=True)
            fit_towers(augmented, splits[region], root / region / "towers", device=args.device)
        return
    if args.stage != "gate":
        from models.t6_router.region_experts import run_experts
        for region in regions:
            print(f"STAGE {args.stage} {region}", flush=True)
            run_experts(workspace, augmented, splits[region], root / region, device=args.device)
        if args.stage == "experts":
            return

    from models.t6_router.bundle import load_bundle
    from models.t6_router.region_gate import run_region_gate
    frames = {}
    inputs = {}
    for region in regions:
        assignment = pd.read_csv(splits[region], usecols=["id", "split", "block"])
        queries = assignment.loc[assignment.split.eq("query")].set_index("id", verify_integrity=True)
        filename = root / region / "experts/predictions.parquet"
        frame = pd.read_parquet(filename).set_index("id", verify_integrity=True)
        if not queries.index.sort_values().equals(frame.index.sort_values()):
            raise ValueError(f"{region}: evaluated IDs differ from the reserved queries")
        if not np.array_equal(queries.loc[frame.index, "block"].astype(str), frame.block.astype(str)):
            raise ValueError(f"{region}: mismatched geographic blocks")
        frames[region] = frame.assign(region=region).reset_index()
        inputs[region] = {"query_photos": len(queries), "blocks": int(queries.block.nunique()),
                          "split_sha256": sha256(splits[region]), "predictions_sha256": sha256(filename)}
    development = pd.concat([frames[key] for key in regions[:-1]], ignore_index=True)
    model = workspace / "data/local_runs/t6_aug8192/bundle_v2/T6.json"
    print("STAGE comparing frozen selectors on final region", flush=True)
    run_region_gate(development, frames["final"], bundle_assets=load_bundle(model), output_dir=root / "gate")
    results = json.loads((root / "gate/summary.json").read_text(encoding="utf-8"))
    boundaries = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    report = {"design": "Three rotating 30 km square development regions and a fourth reserved square. "
                        "A 10 km buffer surrounds each withheld square; the final square and buffer are "
                        "excluded throughout. The official tests are excluded. Each fold has a separately "
                        "fitted T4 and T5, shared by the selectors on that fold.",
              "inputs": {"canonical_split_sha256": sha256(canonical), "feature_ids_sha256": sha256(variants),
                         "baseline_bundle_sha256": sha256(model), "folds": inputs},
              "region_assignment": boundaries["regions"], "results": results}
    temporary = root / "study.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, root / "study.json")
    print("COMPLETE", root / "study.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("workspace", "augmented-root", "output-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=["all", "towers", "experts", "gate"], default="all")
    run(parser.parse_args())
