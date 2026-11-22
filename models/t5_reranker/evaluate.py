"""Compare T5 with the paired T3 year-winner and T1 grid predictions on real queries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.clip_regression.scripts.compare_t1 import (
    EVAL_SPLITS, ERROR_ATOL_KM, _errors, _gps, _metrics, _provenance, _validate_ids,
)

GPS = ("latitude", "longitude")
PREDICTED = ("lat_pred", "lon_pred")


def original_rows(data_root, split_csv):
    """Get exactly the canonical train/evaluation rows with an original cached image."""
    metadata_path = data_root / "meta/smapshot.parquet"
    ids_path = data_root / "features/clip_vitl14_ids.npy"
    meta = pd.read_parquet(metadata_path, columns=["id", *GPS]).set_index("id", verify_integrity=True)
    split = pd.read_csv(split_csv, usecols=["id", "split", "block"]).set_index("id", verify_integrity=True)
    ids = np.load(ids_path, allow_pickle=False)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids <= 0).any() or len(np.unique(ids)) != len(ids):
        raise ValueError("Evaluation requires the original feature cache with one positive integer ID per row")
    rows = split.loc[split.index.isin(ids)]
    rows = rows.loc[rows.split.isin(("train", *EVAL_SPLITS))].join(meta, validate="one_to_one").sort_index()
    _gps(rows, GPS, "metadata")
    truth = rows.loc[rows.split.isin(EVAL_SPLITS)]
    train = rows.loc[rows.split.eq("train")]
    if set(truth.split.unique()) != set(EVAL_SPLITS) or train.empty:
        raise ValueError("Missing canonical training or evaluation split")
    return truth, train, {"metadata": metadata_path, "feature_ids": ids_path, "split": split_csv}


def read_predictions(path, name, truth, train):
    filters = [("method", "=", "year")] if name == "T3" else None
    frame = pd.read_parquet(path, filters=filters)
    _validate_ids(frame, name)
    frame = frame.set_index("id").sort_index()
    if not frame.index.equals(truth.index) or not frame.split.equals(truth.split):
        raise ValueError(f"{name}: IDs or splits differ from the real evaluation query set")
    predicted = _gps(frame, PREDICTED, name)
    if name in ("T3", "T5"):
        ids = frame.neighbor_id
        if not pd.api.types.is_integer_dtype(ids) or ids.isna().any() or not ids.isin(train.index).all():
            raise ValueError(f"{name}: every winning neighbor must be a real training photo with features")
        if "neighbor_split" in frame and not frame.neighbor_split.eq("train").all():
            raise ValueError(f"{name}: invalid neighbor_split")
        if not np.array_equal(predicted, train.loc[ids, list(GPS)].to_numpy(dtype=float)):
            raise ValueError(f"{name}: reported coordinates differ from the winning training photo")
    errors = _errors(truth[list(GPS)].to_numpy(dtype=float), predicted)
    if "error_km" in frame and not np.allclose(frame.error_km, errors, rtol=0, atol=ERROR_ATOL_KM):
        raise ValueError(f"{name}: saved errors differ from recomputed distances")
    return frame, errors


def paired_block_intervals(before, after, blocks):
    """Bootstrap pooled median/accuracy differences with shared spatial block draws."""
    if blocks.isna().any() or blocks.nunique() < 2:
        raise ValueError("test_in needs at least two blocks and a block for each real photo")
    block_values = blocks.to_numpy()
    members = [np.flatnonzero(block_values == block) for block in np.unique(block_values)]
    rng = np.random.default_rng(0)
    draws = np.empty((1000, 2))
    for i in range(len(draws)):
        sampled = rng.integers(len(members), size=len(members))
        rows = np.concatenate([members[j] for j in sampled])
        a, b = before[rows], after[rows]
        draws[i] = np.median(a) - np.median(b), 100 * ((b <= 25).mean() - (a <= 25).mean())
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    return {"seed": 0, "repetitions": len(draws), "blocks": len(members),
            "median_reduction_km_ci95": [float(lo[0]), float(hi[0])],
            "acc25_gain_pp_ci95": [float(lo[1]), float(hi[1])]}


def shortlist_report(path, truth, train, t3, t5):
    """Check both winners against the exact T3 shortlist and quantify its 25-km ceiling."""
    with np.load(path, allow_pickle=False) as cache:
        ids = cache["query_ids"]
        candidates = cache["parent_ids"]
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or candidates.dtype.kind not in "iu":
        raise ValueError(f"{path}: candidate IDs must be integers")
    order = np.argsort(ids)
    if not np.array_equal(ids[order], truth.index.to_numpy()) or candidates.shape != (len(truth), 50):
        raise ValueError(f"{path}: query IDs or top-50 dimensions mismatch")
    candidates = candidates[order]
    if np.any(np.diff(np.sort(candidates, axis=1), axis=1) == 0):
        raise ValueError(f"{path}: duplicate parents in T3 shortlist")
    for name, winner in (("T3", t3), ("T5", t5)):
        if not np.all((candidates == winner.neighbor_id.to_numpy()[:, None]).any(axis=1)):
            raise ValueError(f"{path}: {name} winner missing from T3 shortlist")
    possible = 0
    for start in range(0, len(truth), 2048):
        ref = candidates[start:start + 2048]
        query_latlon = truth.iloc[start:start + len(ref)][list(GPS)].to_numpy(dtype=float)
        reference_latlon = train.loc[ref.ravel(), list(GPS)].to_numpy(dtype=float)
        distances = _errors(np.repeat(query_latlon, ref.shape[1], axis=0), reference_latlon).reshape(ref.shape)
        possible += int((distances <= 25).any(axis=1).sum())
    return {"queries_with_candidate_within_25km": possible,
            "pct_with_candidate_within_25km": 100 * possible / len(truth)}


def evaluate(t5, t3, t1, data_root, split_csv, candidates):
    truth, train, inputs = original_rows(data_root, split_csv)
    inputs.update(T5=t5, T3=t3, T1=t1)
    frames, errors = {}, {}
    for model in ("T3", "T5", "T1"):
        frames[model], errors[model] = read_predictions(inputs[model], model, truth, train)
    by_split = {}
    for split in EVAL_SPLITS:
        mask = truth.split.eq(split).to_numpy()
        real = truth.loc[mask]
        before, after = errors["T3"][mask], errors["T5"][mask]
        a, b = before <= 25, after <= 25
        t3_hits = int(a.sum())
        t5_hits = int(b.sum())
        models = {name: _metrics(values[mask]) for name, values in errors.items()}
        old, new = frames["T3"].loc[mask], frames["T5"].loc[mask]
        row = {"n": len(real), **models,
               "median_reduction_km": models["T3"]["median_km"] - models["T5"]["median_km"],
               "acc25_gain_pp": models["T5"]["acc@25km"] - models["T3"]["acc@25km"],
               "gained_hits": int((~a & b).sum()), "lost_hits": int((a & ~b).sum()),
               "net_hits": t5_hits - t3_hits,
               "changed_winner": int((old.neighbor_id != new.neighbor_id).sum())}
        candidate_file = candidates / f"{split}.npz"
        inputs[f"{split}_candidates"] = candidate_file
        shortlist = shortlist_report(candidate_file, real, train, old, new)
        possible = shortlist["queries_with_candidate_within_25km"]
        shortlist["t3_pct_hits_given_near_candidate"] = 100 * t3_hits / possible if possible else None
        shortlist["t5_pct_hits_given_near_candidate"] = 100 * t5_hits / possible if possible else None
        row["shortlist"] = shortlist
        if split == "test_in":
            row["paired_block_bootstrap"] = paired_block_intervals(before, after, real.block)
        by_split[split] = row
    return {"model": "T5", "paired_baseline": "T3 year-winner", "grid_baseline": "T1",
            "positive_change": "Median T3 minus T5 (km), hit rate T5 minus T3 (percentage points)",
            "earth_radius_km": 6371.0088, "hit_threshold_km": 25,
            "evaluation_queries": len(truth), "training_references": len(train),
            "sorted_query_ids_le64_sha256": hashlib.sha256(truth.index.to_numpy(dtype="<i8").tobytes()).hexdigest(),
            "splits": by_split, "inputs": {name: _provenance(path) for name, path in inputs.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option, help_text in (("t5", "T5 prediction parquet"), ("t3", "T3 year-winner prediction parquet"),
                              ("t1", "T1 grid prediction parquet"), ("data-root", "Original metadata and cached ID root"),
                              ("split-csv", "Canonical splits and blocks"), ("candidates", "T3 top-50 candidate directory"),
                              ("output", "Destination comparison JSON")):
        parser.add_argument("--" + option, required=True, type=Path, help=help_text)
    args = parser.parse_args()
    result = evaluate(args.t5, args.t3, args.t1, args.data_root, args.split_csv, args.candidates)
    output = args.output.resolve()
    if any(output == Path(p["path"]).resolve() for p in result["inputs"].values()):
        parser.error("Output path collides with an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=output.parent) as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
        temporary = Path(out.name)
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    for split, row in result["splits"].items():
        print(f"{split}: n={row['n']}, T3 to T5 median {row['T3']['median_km']:.3f} to {row['T5']['median_km']:.3f} km, "
              f"gained={row['gained_hits']} lost={row['lost_hits']} net={row['net_hits']}")


if __name__ == "__main__":
    main()
