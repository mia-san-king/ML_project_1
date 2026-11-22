"""Auditable geometric ceiling for T3 retrieval and observed hit rates by local density."""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import KDTree

from compare_t3_t4_8192 import (
    ROOT, RUN, EXPECTED, T1_ORIGINAL, T1_AUGMENTED, authoritative_data, checked_errors, source
)

EARTH_KM = 6371.0088
OUT = ROOT / "data/local_runs/t3_refine_analysis"


def unit_xyz(frame):
    lat, lon = np.radians(frame[["latitude", "longitude"]].to_numpy()).T
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def chord(kilometers):
    return 2 * np.sin(kilometers / (2 * EARTH_KM))


def per_group(frame):
    n = len(frame)
    if not n:
        return {"n": 0}
    return {
        "n": n, "distinct_blocks": int(frame.block.nunique()), "median_nearest_km": float(frame.nearest_km.median()),
        "median_train_counts": {c: float(frame[c].median()) for c in ["count_5", "count_25", "coordinates_25"]},
        "max_train_counts": {c: int(frame[c].max()) for c in ["count_5", "count_25"]},
        "oracle_hit_rate_pct": {str(k): float((frame["count_" + str(k)] > 0).mean() * 100) for k in [5, 25]},
        "models": {key.removesuffix("_km"): {
            "hit_25_n": int((frame[key] <= 25).sum()), "hit_25_pct": float((frame[key] <= 25).mean() * 100),
            "median_km": float(frame[key].median()),
        } for key in frame.columns if key.startswith(("t3_", "t1_"))},
    }


def descriptive_rho(frame, density, model):
    # Coarsen by the preassigned evaluation blocks to avoid counting every photo
    # in one location as independent. This is descriptive, not an inferential test.
    grouped = frame.groupby("block").agg(density=(density, "median"), error_km=(model, "median"),
                    hit_pct=(model, lambda errors: 100 * (errors <= 25).mean()))
    return {"individual_n": len(frame), "block_n": len(grouped), "rho_density_error_per_photo": float(
        spearmanr(frame[density], frame[model]).statistic), "rho_block_density_vs_block_error": float(
        spearmanr(grouped.density, grouped.error_km).statistic), "rho_block_density_vs_block_hit_pct": float(
        spearmanr(grouped.density, grouped.hit_pct).statistic)}


def main():
    truth, train, inputs = authoritative_data()
    assert len(train) == 176308 and train.index.min() > 0 and train.index.is_unique
    unique = train.drop_duplicates(["latitude", "longitude"])
    query = unit_xyz(truth)
    tree = KDTree(unit_xyz(train), leaf_size=48)
    unique_tree = KDTree(unit_xyz(unique), leaf_size=48)
    nearest_chord = tree.query(query, k=1, return_distance=True)[0].ravel()
    frame = truth[["split", "block"]].copy()
    frame["nearest_km"] = 2 * EARTH_KM * np.arcsin(nearest_chord / 2)
    for radius in [5, 25]:
        frame["count_" + str(radius)] = tree.query_radius(query, r=chord(radius), count_only=True)
    frame["coordinates_25"] = unique_tree.query_radius(query, r=chord(25), count_only=True)
    assert np.array_equal(frame.nearest_km.to_numpy() <= 25, frame.count_25.to_numpy() > 0)
    pred_paths = {"t3_original": RUN / "original/T3/predictions.parquet",
                  "t3_aug8192": RUN / "augmented/T3/predictions.parquet",
                  "t1_original": T1_ORIGINAL / "results/preds/zhang7_T1_dual_time_available_s0_20261118.parquet",
                  "t1_aug8192": T1_AUGMENTED / "results/preds/T1_aug8192_s0.parquet"}
    for key, path in pred_paths.items():
        error = checked_errors(path, key[:2].upper(), truth, train)
        if key.startswith("t3_") and (error + 1e-7 < frame.nearest_km).any():
            raise ValueError("A discrete retrieval beat the closest actual reference")
        frame[key + "_km"] = error
    results = {
        "metric": "Counts of real T3 reference photos within spherical 5 or 25 km of TRUE query GPS (6371.0088 km). A train label can serve only when a real parent is within the hit radius. Numbers after 8192 variants use the same real parents. coordinates_25 deduplicates exactly identical latitude/longitude tuples.",
        "training_real_photo_ids": len(train), "unique_gps_exact": len(unique),
        "sources": {**inputs, **{k:source(p) for k,p in pred_paths.items()}}, "splits": {}
    }
    boundaries = [0, 1, 100, 1000, 5000, 10000, np.inf]
    for split in EXPECTED:
        sub = frame.loc[frame.split.eq(split)]
        bins = {f"{low}-{high - 1}" if np.isfinite(high) else f"{low}+": per_group(sub.loc[
            sub.count_25.ge(low) & sub.count_25.lt(high)]) for low, high in zip(boundaries[:-1], boundaries[1:])}
        results["splits"][split] = {"all": per_group(sub), "train_photos_within_25km_bins": bins,
            "within_25km_possible": per_group(sub.loc[sub.count_25.gt(0)]),
            "within_25km_impossible": per_group(sub.loc[sub.count_25.eq(0)]),
        }
        if split == "test_in":
            results["splits"][split]["observed_spearman"] = {
                measurement: descriptive_rho(sub, measurement, "t3_aug8192_km")
                for measurement in ["count_5", "count_25", "coordinates_25"]}
    OUT.mkdir(exist_ok=True, parents=True)
    frame.to_parquet(OUT / "density_by_id.parquet")
    (OUT / "density.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    for split, result in results["splits"].items():
        all_ = result["all"]
        print(split, "medNearest, med5, med25, medU25, ceilings, hits", all_["median_nearest_km"], all_["median_train_counts"], all_["oracle_hit_rate_pct"], all_["models"]["t3_aug8192"], flush=True)
        print("covered", result["within_25km_possible"], flush=True)
        print([(k, v["n"], v.get("models", {}).get("t3_aug8192", {}).get("hit_25_pct")) for k, v in result["train_photos_within_25km_bins"].items()], flush=True)
        if "observed_spearman" in result:
            print(result["observed_spearman"], flush=True)


if __name__ == "__main__":
    main()
