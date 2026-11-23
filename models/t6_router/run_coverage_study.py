"""Compare three and seven training regions at the same real-photo budget."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from models.t6_router.expanded_regions import FINAL, SEVEN, matched_samples, write_expanded_splits
from scripts.confidence.evaluate import sha256


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _splits(root, canonical, cache_ids):
    directory = root / "splits"
    kwargs = {"canonical_split": canonical, "augmented_ids": cache_ids}
    if not directory.exists():
        return write_expanded_splits(output_dir=directory, **kwargs)
    with TemporaryDirectory(dir=root) as temporary:
        expected = write_expanded_splits(output_dir=Path(temporary) / "splits", **kwargs)
        saved = {name: directory / path.name for name, path in expected.items()}
        for name, path in expected.items():
            if sha256(saved[name]) != sha256(path):
                raise ValueError(f"{name}: assignments differ from the predeclared region protocol")
        if _json(directory / "manifest.json") != _json(Path(temporary) / "splits/manifest.json"):
            raise ValueError("Regional manifest does not match the regenerated assignments")
    return saved


def read_fold(root, name, split):
    directory = root / name / "experts"
    receipt = _json(directory / "summary.json")
    path = directory / "predictions.parquet"
    if receipt["inputs"]["split_sha256"] != sha256(split) or receipt["predictions_sha256"] != sha256(path):
        raise ValueError(f"{name}: expert predictions do not match their split and receipt")
    rows = pd.read_csv(split, usecols=["id", "split", "block"])
    queries = rows.loc[rows.split.eq("query")].set_index("id", verify_integrity=True).sort_index()
    frame = pd.read_parquet(path).set_index("id", verify_integrity=True).sort_index()
    if not queries.index.equals(frame.index) or len(frame) != receipt["query_count"]:
        raise ValueError(f"{name}: expert query IDs do not match the reserved photos")
    if not frame.region.eq(name).all() or not np.array_equal(queries.block.astype(str), frame.block.astype(str)):
        raise ValueError(f"{name}: unexpected region or block in expert predictions")
    return frame.reset_index(), receipt


def run(workspace, augmented, root, *, device="cuda", stage="all"):
    from models.t6_router.region_towers import fit_towers

    root.mkdir(parents=True, exist_ok=True)
    splits = _splits(root, workspace / "data/splits/main.csv.gz", augmented / "features/clip_vitl14_ids.npy")
    for name in (*SEVEN, FINAL):
        print("STAGE", stage, name, flush=True)
        if stage == "towers":
            fit_towers(augmented, splits[name], root / name / "towers", device=device)
        else:
            from models.t6_router.region_experts import run_experts
            run_experts(workspace, augmented, splits[name], root / name, device=device, refit_rerank=True)
    if stage == "towers":
        return

    from models.t6_router.coverage_gate import USED, _signature, run_coverage_gate
    frames, folds = {}, {}
    for name, path in splits.items():
        frames[name], folds[name] = read_fold(root, name, path)
    final = frames.pop(FINAL)
    arms = matched_samples(pd.concat(frames.values(), ignore_index=True))
    three, seven = arms["three"], arms["seven"]
    overlap = len(set(three.id) & set(seven.id))
    print("STAGE comparing", len(three), len(seven), "shared", overlap, "final", len(final), flush=True)
    output = root / "gate"
    if not output.exists():
        result = run_coverage_gate(three, seven, final, output_dir=output)
    else:
        result = _json(output / "summary.json")
        if result["final_input_sha256"] != _signature(final, USED):
            raise ValueError("Stored routing comparison has different final inputs")
        for name, sample in arms.items():
            if result["training"][name + "_regions"]["sample_sha256"] != _signature(sample, USED):
                raise ValueError(f"Stored routing comparison has different {name}-region inputs")
        for filename, digest in result["outputs_sha256"].items():
            if sha256(output / filename) != digest:
                raise ValueError(f"Unexpected routing artifact: {filename}")

    report = {"design": "New final is the largest of five predeclared new squares by original query count. "
                         "Both groups use the same fold-specific experts and final predictions. "
                         "T5 reranking parameters are refitted on allowed validation data. "
                         "The new and previous final squares, with buffers, are excluded from all fitting.",
              "sampling_seed": 42, "shared_training_ids": overlap,
              "assignment": _json(root / "splits/manifest.json"), "folds": folds, "comparison": result}
    temporary = root / "study.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, root / "study.json")
    print("COMPLETE", root / "study.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("workspace", "augmented", "root"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=["towers", "all"], default="all")
    run(**vars(parser.parse_args()))
