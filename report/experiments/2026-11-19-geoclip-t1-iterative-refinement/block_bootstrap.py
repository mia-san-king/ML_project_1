"""Paired geographic-block bootstrap: median(error at 8) - median(error at 128).

Resample validation blocks, retaining all sampled photos within each block.
The same draws pair both steps and both radii. Positive values mean improvement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze(snapshots, splits, repetitions=1000, seed=0):
    required = {"id", "radius_km", "step", "error_km"}
    if not required.issubset(snapshots.columns):
        raise ValueError(f"Missing snapshot columns: {required - set(snapshots.columns)}")
    if splits.id.duplicated().any():
        raise ValueError("Split table contains duplicate IDs")
    rows = snapshots.loc[snapshots.step.isin([8, 128])].copy()
    if set(rows.radius_km) != {1, 5} or rows.empty:
        raise ValueError("Expected radii 1 and 5 km with steps 8 and 128")
    if rows.duplicated(["id", "radius_km", "step"]).any():
        raise ValueError("Duplicate ID/radius/step snapshot")
    if not np.isfinite(rows.error_km).all() or rows.error_km.lt(0).any():
        raise ValueError("Errors must be finite and nonnegative")
    paired = rows.pivot(index="id", columns=["radius_km", "step"], values="error_km")
    columns = [(radius, step) for radius in (1, 5) for step in (8, 128)]
    if set(paired.columns) != set(columns) or paired.isna().any().any():
        raise ValueError("Every ID must have both steps at both radii")
    geography = splits.set_index("id").reindex(paired.index)
    if geography[["split", "block"]].isna().any().any() or not geography.split.eq("val").all():
        raise ValueError("Every paired ID must map to a validation block")
    block_ids = np.sort(geography.block.unique())
    if len(block_ids) < 2 or repetitions < 1:
        raise ValueError("Need at least two blocks and one repetition")
    members = [np.flatnonzero(geography.block.to_numpy() == block) for block in block_ids]
    errors = paired.loc[:, columns].to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, 2))
    for iteration in range(repetitions):
        sampled_blocks = rng.integers(len(members), size=len(members))
        sampled_rows = np.concatenate([members[index] for index in sampled_blocks])
        medians = np.median(errors[sampled_rows], axis=0)
        draws[iteration] = medians[[0, 2]] - medians[[1, 3]]
    result = {
        "estimand": "median_error_8_km - median_error_128_km; positive means improvement",
        "resampling": "paired blocks with replacement; pooled photo medians, not median paired differences",
        "interval": "95% percentile bootstrap interval",
        "repetitions": repetitions, "seed": seed, "n_photos": len(paired),
        "n_blocks": len(block_ids), "shared_block_draws_across_radii": True,
        "photos_per_block": {str(block): len(indices) for block, indices in zip(block_ids, members)},
        "radii": {},
    }
    observed = np.median(errors, axis=0)
    for index, radius in enumerate((1, 5)):
        result["radii"][str(radius)] = {
            "median_8_km": float(observed[index * 2]),
            "median_128_km": float(observed[index * 2 + 1]),
            "median_reduction_km": float(observed[index * 2] - observed[index * 2 + 1]),
            "percentile_95_km": np.quantile(draws[:, index], [.025, .975]).tolist(),
        }
    return result


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=folder / "snapshots.parquet")
    parser.add_argument("--splits", type=Path, default=folder.parents[1] / "ML_project_1_time/data/splits/main.csv.gz")
    parser.add_argument("--output", type=Path, default=folder / "block_bootstrap.json")
    args = parser.parse_args()
    result = analyze(pd.read_parquet(args.snapshots), pd.read_csv(args.splits))
    result["inputs"] = {"snapshots": str(args.snapshots.resolve()), "splits": str(args.splits.resolve())}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
