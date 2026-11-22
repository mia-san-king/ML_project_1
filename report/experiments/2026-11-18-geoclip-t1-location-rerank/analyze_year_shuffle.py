#!/usr/bin/env python3
"""One-off fixed-candidate test_in year-shuffle diagnostic; no model fitting."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd


SCORE_ATOL = 2e-6  # Report every changed winner, even within this numerical bound.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, result):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def penalties(query_years, reference_years, indices):
    candidate_years = reference_years[indices]
    valid = np.isfinite(query_years[:, None]) & np.isfinite(candidate_years)
    differences = np.zeros(indices.shape, dtype=np.float64)
    np.subtract(query_years[:, None], candidate_years, out=differences, where=valid)
    return np.minimum(np.abs(differences), 50.) / 50.


def winners(indices, scores, penalty, gamma):
    adjusted = scores - gamma * penalty
    columns = adjusted.argmax(axis=1)  # Keep original spatial rank on an exact tie.
    return indices[np.arange(len(indices)), columns], columns, adjusted


def metrics(truth, predicted):
    lat1, lon1 = np.radians(truth).T
    lat2, lon2 = np.radians(predicted).T
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    error = 12742.0176 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    return {"n": len(error), "median_km": float(np.median(error)), "mean_km": float(np.mean(error)),
            "acc@25km": float(100 * np.mean(error <= 25))}


def validate_main(frame, arrays, gamma, penalty):
    method = "location" if gamma == 0 else "year"
    main = frame.loc[frame.split.eq("test_in") & frame.method.eq(method)].set_index("id")
    require(main.index.is_unique, f"{method}: duplicate main query IDs")
    query_ids, reference_ids = arrays["query_ids"], arrays["reference_ids"]
    require(set(main.index) == set(query_ids), f"{method}: main query IDs differ from candidate IDs")
    main = main.loc[query_ids]
    require(main.gamma.eq(gamma).all(), f"{method}: expected fixed gamma={gamma}")
    reference_lookup = {int(identifier): index for index, identifier in enumerate(reference_ids)}
    require(main.neighbor_id.isin(reference_ids).all(), f"{method}: main winner is outside train references")
    main_indices = np.array([reference_lookup[int(identifier)] for identifier in main.neighbor_id], dtype=np.int64)
    matches = arrays["indices"] == main_indices[:, None]
    present = matches.any(axis=1)
    main_columns = matches.argmax(axis=1)
    expected, columns, adjusted = winners(arrays["indices"], arrays["scores"], penalty, gamma)
    rows = np.arange(len(expected))
    np.testing.assert_allclose(main[["lat_pred", "lon_pred"]], arrays["reference_gps"][main_indices], rtol=0, atol=1e-8,
                               err_msg=f"{method}: main neighbor ID/GPS mismatch")
    main_penalty = penalties(arrays["query_years"], arrays["reference_years"], main_indices[:, None])[:, 0]
    for column, values, atol in (
        ("score", main.spatial_cosine.to_numpy() - gamma * main_penalty, 1e-10),
        ("year_penalty", main_penalty, 1e-10),
        ("query_year", arrays["query_years"], 1e-10),
        ("neighbor_year", arrays["reference_years"][main_indices], 1e-10),
    ):
        np.testing.assert_allclose(main[column], values, rtol=0, atol=atol, equal_nan=True, err_msg=f"{method}: {column}")
    changed = np.flatnonzero(expected != main_indices)
    mismatches = []
    for i in changed:
        gap = float(adjusted[i, columns[i]] - adjusted[i, main_columns[i]]) if present[i] else None
        mismatches.append({"query_id": int(query_ids[i]), "main_neighbor_id": int(reference_ids[main_indices[i]]),
                           "recomputed_neighbor_id": int(reference_ids[expected[i]]), "score_gap": gap,
                           "classification": "main_winner_outside_candidate_list" if gap is None else
                           ("numerical_near_tie" if gap <= SCORE_ATOL else "substantive")})
    score_difference = np.abs(main.spatial_cosine.to_numpy()[present] - arrays["scores"][rows[present], main_columns[present]])
    return {"queries": len(main), "winner_mismatches": len(mismatches), "mismatches": mismatches,
            "exact_winner_match": not mismatches,
            "main_winner_cosine_max_abs_difference": float(score_difference.max()) if len(score_difference) else None,
            "main_winner_cosine_differences_above_tolerance": int((score_difference > SCORE_ATOL).sum())}, expected, main_indices


def analyze(candidates, predictions, output, *, allow_cpu_paired_sensitivity=False):
    candidates, predictions, output = map(Path, (candidates, predictions, output))
    require(not output.exists(), f"output already exists: {output}")
    result = {"status": "validating", "split": "test_in", "fixed_gamma": 1., "seeds": list(range(20)),
              "inputs": {"candidates": str(candidates.resolve()), "predictions": str(predictions.resolve()),
                         "candidates_sha256": sha256(candidates), "predictions_sha256": sha256(predictions)},
              "score_validation_atol": SCORE_ATOL,
              "scope": "fixed test_in candidates; post-hoc diagnostic, no tuning or new model"}
    try:
        with np.load(candidates, allow_pickle=False) as archive:
            keys = ("query_ids", "reference_ids", "indices", "scores", "query_years", "reference_years", "query_gps", "reference_gps")
            arrays = {key: archive[key] for key in keys}
        n, r = len(arrays["query_ids"]), len(arrays["reference_ids"])
        for key, count in (("query_ids", n), ("reference_ids", r)):
            require(arrays[key].shape == (count,) and np.issubdtype(arrays[key].dtype, np.integer), f"invalid {key}")
            require(len(np.unique(arrays[key])) == count, f"duplicate {key}")
        require(not np.intersect1d(arrays["query_ids"], arrays["reference_ids"]).size, "query/reference IDs overlap")
        indices, scores = arrays["indices"], arrays["scores"].astype(np.float64)
        arrays["scores"] = scores
        require(n > 0 and r > 0 and indices.ndim == 2 and indices.shape[0] == n and indices.shape[1] > 0, "empty or invalid candidates")
        require(np.issubdtype(indices.dtype, np.integer) and ((indices >= 0) & (indices < r)).all(), "invalid reference indices")
        require(scores.shape == indices.shape and np.isfinite(scores).all() and (np.diff(scores, axis=1) <= 0).all(), "invalid sorted cosine scores")
        require((np.diff(np.sort(indices, axis=1), axis=1) != 0).all(), "duplicate candidate reference within a query")
        for prefix, count in (("query", n), ("reference", r)):
            gps, years = arrays[prefix + "_gps"], arrays[prefix + "_years"].astype(np.float64)
            arrays[prefix + "_years"] = years
            require(gps.shape == (count, 2) and np.isfinite(gps).all(), f"invalid {prefix} GPS")
            require((np.abs(gps[:, 0]) <= 90).all() and (np.abs(gps[:, 1]) <= 180).all(), f"invalid {prefix} GPS bounds")
            require(years.shape == (count,) and not np.isinf(years).any(), f"invalid {prefix} years")
            require(((years[np.isfinite(years)] >= 1800) & (years[np.isfinite(years)] <= 2030)).all(), f"invalid {prefix} year bounds")
        original_penalty = penalties(arrays["query_years"], arrays["reference_years"], indices)
        frame = pd.read_parquet(predictions)
        result["validation"], original_winners, main_winners = {}, {}, {}
        for method, gamma in (("location", 0.), ("year", 1.)):
            report, selected, main_selected = validate_main(frame, arrays, gamma, original_penalty)
            result["validation"][method], original_winners[method] = report, selected
            main_winners[method] = main_selected
        disagreement = any(report["winner_mismatches"] or report["main_winner_cosine_differences_above_tolerance"] for report in result["validation"].values())
        require(not disagreement or allow_cpu_paired_sensitivity,
                "main run does not reproduce exactly; inspect differences before explicitly allowing CPU paired sensitivity")
    except (ValueError, KeyError, AssertionError) as exc:
        result.update(status="validation_failed", error=str(exc))
        save(output, result)
        raise ValueError(f"candidate validation failed; details saved to {output}: {exc}") from exc

    known_reference = np.isfinite(arrays["reference_years"])
    known_query = np.isfinite(arrays["query_years"])
    result["original"] = {method: metrics(arrays["query_gps"], arrays["reference_gps"][selected]) for method, selected in original_winners.items()}
    result["original_definition"] = "true archive years evaluated on the fixed diagnostic candidate artifact"
    result["original_main"] = {method: metrics(arrays["query_gps"], arrays["reference_gps"][selected]) for method, selected in main_winners.items()}
    result["known_query_missing_reference"] = {}
    for method, selected in main_winners.items():
        count = int((known_query & ~known_reference[selected]).sum())
        result["known_query_missing_reference"][method] = {"known_queries": int(known_query.sum()), "count": count,
                                                            "percent": float(count / known_query.sum() * 100) if known_query.any() else None}
    result["shuffle_policy"] = {"gamma": 1., "permuted_values": "all finite reference years across the full train reference pool",
                                 "nan_mask_preserved": True, "query_years_fixed": True,
                                 "candidate_indices_and_cosines_fixed": True, "numpy_generator": "default_rng (PCG64)",
                                 "known_reference_count": int(known_reference.sum()), "missing_reference_count": int((~known_reference).sum())}
    result["shuffles"] = []
    for seed in range(20):
        shuffled = arrays["reference_years"].copy()
        shuffled[known_reference] = np.random.default_rng(seed).permutation(shuffled[known_reference])
        require(np.array_equal(np.isnan(shuffled), np.isnan(arrays["reference_years"])), "shuffle changed missing mask")
        penalty = penalties(arrays["query_years"], shuffled, indices)
        selected, _, _ = winners(indices, scores, penalty, 1.)
        row = {"seed": seed, **metrics(arrays["query_gps"], arrays["reference_gps"][selected])}
        result["shuffles"].append(row)
        print(f"seed={seed:02d} median_km={row['median_km']:.6f} acc@25km={row['acc@25km']:.6f}", flush=True)
    result["shuffle_distribution"] = {
        key: {"median": float(np.median([row[key] for row in result["shuffles"]])),
              "min": float(min(row[key] for row in result["shuffles"])), "max": float(max(row[key] for row in result["shuffles"]))}
        for key in ("median_km", "acc@25km")}
    mismatches = sum(report["winner_mismatches"] for report in result["validation"].values())
    result["status"] = "complete_cpu_paired_sensitivity" if allow_cpu_paired_sensitivity else "complete"
    if allow_cpu_paired_sensitivity:
        result["scope"] = "explicit CPU paired sensitivity: fixed CPU candidates and gamma, true versus shuffled years; not claimed to reproduce the original GPU run"
    result["source_sha256"] = sha256(__file__)
    save(output, result)
    print(f"validation winner mismatches={mismatches}; wrote {output}", flush=True)
    return result


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=root / "candidate_diagnostic.npz")
    parser.add_argument("--predictions", type=Path, default=root / "results/predictions.parquet")
    parser.add_argument("--output", type=Path, default=root / "shuffle_diagnostic.json")
    parser.add_argument("--allow-cpu-paired-sensitivity", action="store_true",
                        help="explicitly allow reported CPU/main candidate differences after review; labels result as a separate sensitivity test")
    args = parser.parse_args()
    try:
        analyze(args.candidates, args.predictions, args.output, allow_cpu_paired_sensitivity=args.allow_cpu_paired_sensitivity)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
