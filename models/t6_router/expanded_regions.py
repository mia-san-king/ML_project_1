"""Predeclared regional folds and matched-size development samples for T6.

The new regions were chosen using only canonical split membership and photo
counts. Cores span 3 by 3 LV95 blocks (30 km square), with >=7 development
blocks containing photos and no official evaluation or buffer photos. All
five new windows, including their one-block rings, are disjoint from one
another and from the previous four windows. The most populous of the five
new windows is reserved for final evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from models.geoclip_time.scripts.rerank_references import _sha256
from models.t6_router.region_splits import REGIONS, _inside, _parent_rows


THREE = ("dev1", "dev2", "dev3")
SEVEN = (*THREE, "dev4", "dev5", "dev6", "dev7")
FINAL = "new_final"
CORNERS = {**{name: REGIONS[name] for name in THREE},
           "dev4": (271, 119), "dev5": (259, 122), "dev6": (252, 116),
           "dev7": (249, 111), FINAL: (257, 114)}


def write_expanded_splits(*, canonical_split, augmented_ids, output_dir):
    """Write eight caches of assignments. Both study arms share each expert fold."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new splits directory: {output}")
    cache = np.load(augmented_ids, allow_pickle=False) if isinstance(augmented_ids, (str, Path)) else augmented_ids
    canonical = pd.read_csv(canonical_split, usecols=["id", "split", "block"],
                            dtype={"id": "int64", "block": "int64", "split": str})
    canonical_blocks = canonical.block.to_numpy()
    ids, parents = _parent_rows(canonical, cache)
    blocks = parents.block.to_numpy()
    reserved = _inside(blocks, REGIONS["final"], padding=1) | _inside(blocks, CORNERS[FINAL], padding=1)
    train = parents.split.eq("train").to_numpy()
    val = parents.split.eq("val").to_numpy()
    real = ids > 0
    manifest = {
        "canonical_split_sha256": _sha256(Path(canonical_split)), "roles": {
            "train": "Original train parents and their cached variants outside both reserved squares, the current query square, and all three exclusion rings.",
            "val": "Original positive validation IDs outside the same areas, for expert checkpoint selection.",
            "query": "Real train or validation photo inside the current square, used for predictions only.",
            "ignore": "All other rows, including official tests and all removed parents' variants.",
        },
        "three": list(THREE), "seven": list(SEVEN), "final": FINAL,
        "archived_final_excluded": {"southwest_lv95_m": [i*10_000 for i in REGIONS["final"]],
                                    "side_km": 30, "buffer_km": 10},
        "regions": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "splits"
        stage.mkdir()
        for name, corner in CORNERS.items():
            original = canonical.loc[_inside(canonical_blocks, corner), "split"]
            if original.isin(["test_city", "test_region", "test_in", "buffer"]).any():
                raise ValueError(f"Official evaluation or buffer photos inside predeclared {name}")
            core = _inside(blocks, corner)
            excluded = reserved | _inside(blocks, corner, padding=1)
            query = core & real & (train | val)
            fit = ~excluded & train
            early_stop = ~excluded & real & val
            if not (query.any() and fit.any() and early_stop.any()):
                raise ValueError(f"Empty expert training, early stopping, or query fold: {name}")
            fold = pd.DataFrame({
                "id": ids, "block": blocks, "region": np.where(query, name, ""),
                "split": np.select([fit, early_stop, query], ["train", "val", "query"], default="ignore"),
            })
            path = stage / f"{name}.csv.gz"
            fold.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            manifest["regions"][name] = {
                "southwest_lv95_m": [i*10_000 for i in corner], "side_km": 30, "buffer_km": 10,
                "canonical_real_queries": int(original.isin(["train", "val"]).sum()),
                "canonical_train": int(original.eq("train").sum()), "canonical_val": int(original.eq("val").sum()),
                "query": int(query.sum()), "blocks": int(np.unique(blocks[query]).size),
                "train_real": int((fit & real).sum()), "train_variant": int((fit & ~real).sum()),
                "val": int(early_stop.sum()), "ignore": int((~(fit | early_stop | query)).sum()),
                "split_sha256": _sha256(path),
            }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return {name: output / f"{name}.csv.gz" for name in CORNERS}


def _quotas(capacities, budget):
    """Water-fill to a common integer cap, redistributing slots from small regions."""
    if budget > sum(capacities.values()) or budget < len(capacities):
        raise ValueError("Sample budget must be feasible and cover every region")
    low, high = 0, budget
    while low < high:
        k = (low + high + 1) // 2
        if sum(min(k, n) for n in capacities.values()) <= budget:
            low = k
        else:
            high = k - 1
    quotas = {name: min(n, low) for name, n in capacities.items()}
    residual = budget - sum(quotas.values())
    for name in [name for name in quotas if quotas[name] < capacities[name]][:residual]:
        quotas[name] += 1
    return quotas


def matched_samples(development, *, budget=46_235, seed=42):
    """Two equally sized photo sets; draw once per region, then reuse its prefix."""
    if not development.columns.is_unique or not {"region", "id"}.issubset(development):
        raise ValueError("Expected a frame with distinct id and region columns")
    if development.region.isna().any() or set(development.region) != set(SEVEN):
        raise ValueError("Expected exactly the seven predeclared development regions")
    if (not pd.api.types.is_integer_dtype(development.id) or development.id.isna().any()
            or development.id.duplicated().any() or (development.id <= 0).any()):
        raise ValueError("Expected unique positive integer photo IDs")
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise ValueError("Expected an integer sample budget")
    draws = {}
    for i, name in enumerate(SEVEN):
        ordered = development.loc[development.region.eq(name)].sort_values("id")
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        draws[name] = ordered.iloc[rng.permutation(len(ordered))]
    matched = {}
    for label, regions in (("three", THREE), ("seven", SEVEN)):
        quotas = _quotas({name: len(draws[name]) for name in regions}, budget)
        rows = [draws[name].iloc[:n] for name, n in quotas.items()]
        matched[label] = pd.concat(rows).sort_values("id").reset_index(drop=True)
    return matched


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for opt in ("canonical-split", "augmented-ids", "output-dir"):
        cli.add_argument("--" + opt, type=Path, required=True)
    args = cli.parse_args()
    write_expanded_splits(**vars(args))
    print((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
