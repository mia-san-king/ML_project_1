"""Predeclared contiguous development holdouts in the official Swiss LV95 grid.

Coordinates are southwest block indices, in units of 10 km. Windows have
three blocks per side and a one-block exclusion ring (also at the corners).
We chose windows from canonical split counts, before computing predictions.
Eligible windows have development photos in >=7 blocks and no regular official
test or buffer rows in the core. The final window maximizes development count
among those whose exclusion ring and core contain no official validation rows.
Each subsequent development window maximizes count subject to disjoint rings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from models.geoclip_time.lib.temporal_data import _validated_ids
from models.geoclip_time.scripts.rerank_references import _sha256


REGIONS = {"dev1": (266, 122), "dev2": (271, 125), "dev3": (261, 109), "final": (278, 113)}
SIDE = 3


def _inside(blocks, corner, padding=0):
    x, y = blocks // 100_000, blocks % 100_000
    left, bottom = corner
    return ((x >= left - padding) & (x < left + SIDE + padding) &
            (y >= bottom - padding) & (y < bottom + SIDE + padding))


def _parent_rows(canonical, cache_ids):
    ids = _validated_ids(cache_ids, "augmented feature cache")
    if (ids == 0).any() or (ids == np.iinfo(np.int64).min).any():
        raise ValueError("Expected nonzero IDs identifying real photo parents")
    parents = np.abs(ids)
    if not np.isin(parents[ids < 0], ids[ids > 0]).all():
        raise ValueError("A diffusion variant has no real cached parent")
    by_id = canonical.set_index("id", verify_integrity=True)
    if not np.isin(parents, by_id.index).all() or (by_id.index <= 0).any():
        raise ValueError("Use the official real-photo split for all cached parents")
    rows = by_id.loc[parents].reset_index(drop=True)
    if not rows.loc[ids < 0, "split"].eq("train").all():
        raise ValueError("A diffusion parent is outside the original train split")
    return ids, rows


def write_region_splits(*, canonical_split, augmented_ids, output_dir):
    """Write all cache IDs in cache order. A variant always inherits its parent block."""
    canonical_split, output_dir = Path(canonical_split), Path(output_dir)
    cache = np.load(augmented_ids, allow_pickle=False) if isinstance(augmented_ids, (Path, str)) else augmented_ids
    canonical = pd.read_csv(canonical_split, usecols=["id", "block", "split"],
                            dtype={"id": "int64", "block": "int64", "split": str})
    canonical_blocks = canonical.block.to_numpy()
    if (canonical.split.eq("val") & _inside(canonical_blocks, REGIONS["final"], padding=1)).any():
        raise ValueError("An official validation query overlaps the final region or its buffer")
    ids, parents = _parent_rows(canonical, cache)
    blocks = parents.block.to_numpy()
    final_buffer = _inside(blocks, REGIONS["final"], padding=1)
    original_train = parents.split.eq("train").to_numpy()
    original_val = parents.split.eq("val").to_numpy()
    real = ids > 0
    records = {}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new splits directory: {output_dir}")
    with TemporaryDirectory(dir=output_dir.parent) as temp:
        stage = Path(temp) / "splits"
        stage.mkdir()
        for name, corner in REGIONS.items():
            core = _inside(blocks, corner)
            excluded = final_buffer | _inside(blocks, corner, padding=1)
            query = core & real & (original_train | original_val)
            fit = original_train & ~excluded
            early_stop = original_val & real & ~excluded
            if not (fit.any() and early_stop.any() and query.any()):
                raise ValueError(f"Empty training, early stopping or query split in {name}")
            fold = pd.DataFrame({"id": ids, "split": np.select([fit, early_stop, query],
                                ["train", "val", "query"], default="ignore"), "block": blocks,
                                "region": np.where(query, name, "")})
            path = stage / f"{name}.csv.gz"
            fold.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            records[name] = {
                "southwest_lv95_m": [int(i * 10_000) for i in corner], "side_km": 30, "buffer_km": 10,
                "canonical_real_queries": int((canonical.split.isin(["train", "val"]) &
                                               _inside(canonical_blocks, corner)).sum()),
                "query": int(query.sum()), "train_real": int((fit & real).sum()),
                "train_variant": int((fit & ~real).sum()), "val": int(early_stop.sum()),
                "split_sha256": _sha256(path),
            }
        info = {"canonical_split_sha256": _sha256(canonical_split), "regions": records}
        (stage / "manifest.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        stage.rename(output_dir)
    return {name: output_dir / f"{name}.csv.gz" for name in REGIONS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("canonical-split", "augmented-ids", "output-dir"):
        parser.add_argument("--" + argument, required=True, type=Path)
    args = parser.parse_args()
    write_region_splits(**vars(args))
    print((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
