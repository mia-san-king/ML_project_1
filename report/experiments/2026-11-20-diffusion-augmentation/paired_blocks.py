"""Paired spatial-block bootstrap of augmentation gains on original test_in queries.

Use --output - for JSON on stdout. All reported improvements are positive:
median error before-minus-after (km), accuracy after-minus-before (percentage points).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_augmented import error_km, metrics, read_predictions

MODELS = ("T1", "T3", "T4")
REPETITIONS = 1000
SEED = 0


def load_errors(args):
    """Reuse compare_augmented's exact baseline files and error calculation."""
    root = Path(__file__).resolve().parent.parent
    original = {
        "T1": root / "2026-11-18-geoclip-time/preds/zhang7_T1_dual_time_available_s0_20261118.parquet",
        "T3": root / "2026-11-18-geoclip-t0-photo-rerank/results/predictions.parquet",
        "T4": root / "2026-11-20-clip-regression-t4/predictions.parquet",
    }
    truth = read_predictions(original["T4"], "T4")
    selected = truth.split.eq("test_in")
    query_truth = truth.loc[selected]
    if query_truth.empty:
        raise ValueError("No original test_in queries")
    errors, paths = [], {}
    for model in MODELS:
        augmented = Path(getattr(args, model.lower()))
        before = read_predictions(original[model], model)
        after = read_predictions(augmented, model)
        for frame in (before, after):
            pd.testing.assert_index_equal(frame.index, truth.index)
            if not np.array_equal(frame.split, truth.split):
                raise ValueError(f"{model}: per-ID split mismatch")
        errors.append(np.column_stack((
            error_km(before.loc[selected], query_truth),
            error_km(after.loc[selected], query_truth),
        )))
        paths[model] = {"before": str(original[model]), "after": str(augmented.resolve())}
    errors = np.stack(errors, axis=1)  # [photo, model, before/after]
    if not np.isfinite(errors).all():
        raise ValueError("All geographic errors must be finite")

    split_path = root.parent / "ML_project_1_time/data/splits/main.csv.gz"
    splits = pd.read_csv(split_path).set_index("id", verify_integrity=True)
    geography = splits.reindex(query_truth.index)
    if geography[["split", "block"]].isna().any().any() or not geography.split.eq("test_in").all():
        raise ValueError("Every evaluated ID must map to its canonical test_in block")
    if not np.isfinite(geography.block.to_numpy(dtype=float)).all():
        raise ValueError("Block IDs must be finite")
    return errors, geography.block.to_numpy(), paths, split_path


def bootstrap(errors, blocks):
    """Draw whole blocks jointly, retaining within-block photo weights and pairs."""
    block_ids = np.unique(blocks)
    if len(block_ids) < 2:
        raise ValueError("At least two spatial blocks are required")
    members = [np.flatnonzero(blocks == block) for block in block_ids]
    rng = np.random.default_rng(SEED)
    reductions = np.empty((REPETITIONS, len(MODELS)))
    accuracy_gains = np.empty_like(reductions)
    for iteration in range(REPETITIONS):
        draw = rng.integers(len(members), size=len(members))
        rows = np.concatenate([members[index] for index in draw])
        sample = errors[rows]
        medians = np.median(sample, axis=0)
        accuracies = (sample <= 25).mean(axis=0) * 100
        reductions[iteration] = medians[:, 0] - medians[:, 1]
        accuracy_gains[iteration] = accuracies[:, 1] - accuracies[:, 0]
    results = {}
    for index, model in enumerate(MODELS):
        before, after = metrics(errors[:, index, 0]), metrics(errors[:, index, 1])
        results[model] = {
            "before": before, "after": after,
            "median_reduction_km": before["median_km"] - after["median_km"],
            "median_reduction_95pct_km": np.quantile(reductions[:, index], [.025, .975]).tolist(),
            "acc25_gain_percentage_points": after["pct25"] - before["pct25"],
            "acc25_gain_95pct_percentage_points": np.quantile(accuracy_gains[:, index], [.025, .975]).tolist(),
        }
    return {"split": "test_in", "n_photos": len(errors), "n_blocks": len(block_ids),
            "repetitions": REPETITIONS, "seed": SEED,
            "interval": "95% percentile paired spatial-block bootstrap",
            "resampling": "Blocks sampled uniformly with replacement; all photos retained; same draws for all models and before/after",
            "positive_means_improvement": True,
            "median_estimand": "Difference of pooled photo medians, not median paired differences",
            "models": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("t1", "t3", "t4", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    errors, blocks, paths, split_path = load_errors(args)
    result = bootstrap(errors, blocks)
    result["inputs"] = {"predictions": paths, "split_csv": str(split_path)}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if str(args.output) == "-":
        print(text, end="")
    else:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
