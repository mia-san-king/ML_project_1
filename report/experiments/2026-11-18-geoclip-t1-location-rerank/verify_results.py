"""Independently verify saved T1-location reranking results; never fit a model."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd


SPLITS = ("val", "test_in", "test_city", "test_region")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, atol=1e-8):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, equal_nan=True, err_msg=label)


def ids_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def calendar_years(metadata):
    """Independent scalar implementation of the recorded f019e23 date policy."""
    values = np.full(len(metadata), np.nan, dtype=float)
    for index, (lower, upper) in enumerate(metadata[["date_shot_min", "date_shot_max"]].itertuples(index=False, name=None)):
        first = pd.to_datetime(lower, errors="coerce", utc=True)
        last = pd.to_datetime(upper, errors="coerce", utc=True)
        if pd.isna(first) or pd.isna(last):
            continue
        midpoint = first + (last - first) / 2
        year = float(midpoint.year) + (midpoint.dayofyear - 1) / 366.0
        if 1800 <= year <= 2030:
            values[index] = year
    return pd.Series(values, index=metadata.index)


def errors_km(frame, metadata):
    truth = metadata.loc[frame.index, ["latitude", "longitude"]].to_numpy(dtype=np.float64)
    predicted = frame[["lat_pred", "lon_pred"]].to_numpy(dtype=np.float64)
    require(np.isfinite(truth).all() and np.isfinite(predicted).all(), "nonfinite coordinates")
    require((np.abs(predicted[:, 0]) <= 90).all() and (np.abs(predicted[:, 1]) <= 180).all(), "invalid predicted GPS")
    lat1, lon1 = np.radians(truth).T
    lat2, lon2 = np.radians(predicted).T
    half_chord = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    half_chord = np.clip(half_chord, 0, 1)
    return 12742.0176 * np.arctan2(np.sqrt(half_chord), np.sqrt(1 - half_chord))


def describe(frame, expected_ids, split_labels, metadata):
    frame = frame.set_index("id", verify_integrity=True).sort_index()
    require(frame.index.equals(expected_ids), "evaluation query IDs differ from available cache/splits")
    require(np.array_equal(frame["split"].to_numpy(), split_labels.loc[frame.index].to_numpy()), "query split mismatch")
    frame["verified_error_km"] = errors_km(frame, metadata)
    if "error_km" in frame:
        close(frame.error_km, frame.verified_error_km, "saved per-query errors", atol=1e-7)
    metrics = {}
    for name in SPLITS:
        error = frame.loc[frame.split.eq(name), "verified_error_km"].to_numpy()
        require(len(error) > 0, f"empty evaluation split: {name}")
        metrics[name] = {"n": len(error), "median_km": float(np.median(error)),
                         "mean_km": float(np.mean(error)), "acc@25km": float(np.mean(error <= 25) * 100)}
    return frame, metrics


def paired(before, after, valid_date):
    output = {}
    for name in SPLITS:
        first = before.loc[before.split.eq(name)]
        second = after.loc[first.index]
        changed = (first[["lat_pred", "lon_pred"]].to_numpy() != second[["lat_pred", "lon_pred"]].to_numpy()).any(axis=1)
        difference = second.verified_error_km.to_numpy() - first.verified_error_km.to_numpy()
        before25 = first.verified_error_km.to_numpy() <= 25
        after25 = second.verified_error_km.to_numpy() <= 25
        unknown = ~valid_date.loc[first.index].to_numpy()
        row = {"n": len(first), "changed_gps": int(changed.sum()), "closer": int((difference < -1e-9).sum()),
               "farther": int((difference > 1e-9).sum()), "equal_error": int((np.abs(difference) <= 1e-9).sum()),
               "within25_before": int(before25.sum()), "within25_after": int(after25.sum()),
               "within25_gained": int((~before25 & after25).sum()), "within25_lost": int((before25 & ~after25).sum()),
               "acc25_delta_pp": float(100 * (after25.mean() - before25.mean())),
               "no_valid_query_date": {"n": int(unknown.sum()), "changed_gps": int(changed[unknown].sum()),
                                       "closer": int((difference[unknown] < -1e-9).sum()),
                                       "farther": int((difference[unknown] > 1e-9).sum())}}
        if "neighbor_id" in first and "neighbor_id" in second:
            changed_reference = first.neighbor_id.to_numpy() != second.neighbor_id.to_numpy()
            row["changed_reference"] = int(changed_reference.sum())
            row["no_valid_query_date"]["changed_reference"] = int(changed_reference[unknown].sum())
        output[name] = row
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent)
    for option in ("data-root", "split-csv", "grid-t1", "grid-t2", "t3"):
        parser.add_argument("--" + option, type=Path)
    args = parser.parse_args(argv)
    run = args.run_dir.resolve()
    base = run.parents[1]
    data = args.data_root or base / "data"
    split_path = args.split_csv or base / "ML_project_1/data/splits/main.csv.gz"
    results = run / "results"
    summary = json.loads((results / "summary.json").read_text())
    selection = json.loads((results / "val_gamma_search.json").read_text())
    require(selection["selection_split"] == "val", "gamma search must use val only")
    require(summary["test_gamma_search"] is False, "test-set gamma selection reported")
    curve = selection["curve"]
    require(len(curve) > 0 and all(np.isfinite(row["median_km"]) for row in curve), "invalid validation curve")
    gamma = float(selection["selected_gamma"])
    require(np.isfinite(gamma) and gamma >= 0, "invalid selected gamma")
    require(gamma == min(curve, key=lambda row: row["median_km"])["gamma"], "selected gamma disagrees with recorded val criterion")
    require(gamma == summary["selected_gamma"], "summary gamma mismatch")
    require([row["gamma"] for row in curve] == selection["gammas"] == summary["gammas"], "gamma grid mismatch")

    metadata = pd.read_parquet(data / "meta/smapshot.parquet", columns=["id", "latitude", "longitude", "date_shot_min", "date_shot_max"]).set_index("id", verify_integrity=True)
    labels = pd.read_csv(split_path).set_index("id", verify_integrity=True)["split"]
    cache_ids = np.load(data / "features/clip_vitl14_ids.npy", allow_pickle=False)
    require(cache_ids.ndim == 1 and len(np.unique(cache_ids)) == len(cache_ids), "cache IDs must be unique")
    cached_labels = labels.loc[cache_ids].to_numpy()
    train_ids = cache_ids[cached_labels == "train"]
    expected_ids = pd.Index(np.sort(cache_ids[np.isin(cached_labels, SPLITS)]), name="id")
    counts = {name: int((labels.loc[expected_ids] == name).sum()) for name in SPLITS}
    require(selection["selection_rows"] == counts["val"], "gamma search did not use full available val")
    require(all(row["n"] == counts["val"] for row in curve), "validation curve count mismatch")
    require(summary["references"]["split"] == "train" and summary["references"]["n"] == len(train_ids), "reference population mismatch")
    require(summary["references"]["ordered_ids_sha256"] == ids_hash(train_ids), "ordered training reference IDs mismatch")
    predictions = pd.read_parquet(results / "predictions.parquet")
    require(set(predictions.method) == {"location", "year"}, "expected location and year methods")
    used_ids = np.unique(np.concatenate([expected_ids.to_numpy(), predictions.neighbor_id.to_numpy()]))
    print(f"checking original metadata dates for {len(used_ids)} query/reference IDs", flush=True)
    years = calendar_years(metadata.loc[used_ids])
    full, metrics = {}, {}
    for method, group in predictions.groupby("method", sort=False):
        require(group.neighbor_id.isin(train_ids).all() and group.neighbor_split.eq("train").all(), "nontraining reference selected")
        close(group[["lat_pred", "lon_pred"]].to_numpy(), metadata.loc[group.neighbor_id, ["latitude", "longitude"]].to_numpy(dtype=float), "neighbor ID/GPS agreement", atol=0)
        require(group.gamma.eq(0.0 if method == "location" else gamma).all(), "row gamma mismatch")
        close(group.query_year, years.loc[group.id], "query dates from original metadata", atol=1e-10)
        close(group.neighbor_year, years.loc[group.neighbor_id], "neighbor dates from original metadata", atol=1e-10)
        valid = np.isfinite(group.query_year.to_numpy()) & np.isfinite(group.neighbor_year.to_numpy())
        penalty = np.zeros(len(group))
        penalty[valid] = np.minimum(np.abs(group.query_year.to_numpy()[valid] - group.neighbor_year.to_numpy()[valid]), 50) / 50
        close(group.year_penalty, penalty, "capped normalized year difference")
        require(np.isfinite(group.spatial_cosine).all() and group.spatial_cosine.abs().le(1 + 1e-5).all(), "invalid unscaled spatial cosine")
        close(group.score, group.spatial_cosine - group.gamma * penalty, "reranked score equation")
        full[method], metrics[method] = describe(group, expected_ids, labels, metadata)

    first, second = full["location"], full["year"]
    require((first.spatial_cosine >= second.spatial_cosine - 1e-7).all(), "location baseline is not highest spatial score")
    require((second.score >= first.spatial_cosine - gamma * first.year_penalty - 1e-7).all(), "reranked winner loses to the baseline candidate under the selected gamma")
    unknown = ~np.isfinite(years.loc[expected_ids].to_numpy())
    require(np.array_equal(first.neighbor_id.to_numpy()[unknown], second.neighbor_id.to_numpy()[unknown]), "missing query dates changed the reference")
    if gamma == 0:
        require(np.array_equal(first.neighbor_id, second.neighbor_id), "gamma zero changed the reference")
    reported = {(row["method"], row["split"]): row for row in summary["metrics"]}
    require(len(summary["metrics"]) == len(reported) == 8, "summary must contain exactly four splits by two methods")
    for method in ("location", "year"):
        for name in SPLITS:
            row = reported[(method, name)]
            require(row["gamma"] == (0.0 if method == "location" else gamma), "metric gamma mismatch")
            for key, value in metrics[method][name].items():
                close(row[key], value, f"summary {method}/{name}/{key}", atol=1e-7)
    for candidate_gamma, method in ((0.0, "location"), (gamma, "year")):
        selected_rows = [row for row in curve if row["gamma"] == candidate_gamma]
        require(len(selected_rows) > 0, "curve lacks baseline or selected gamma")
        for row in selected_rows:
            for key, value in metrics[method]["val"].items():
                close(row[key], value, f"val curve {candidate_gamma}/{key}", atol=1e-7)
    for name in SPLITS:
        query_ids = cache_ids[cached_labels == name]
        require(summary["queries"][name]["n"] == counts[name], "summary query count mismatch")
        require(summary["queries"][name]["ordered_ids_sha256"] == ids_hash(query_ids), "summary ordered query IDs mismatch")
        require(summary["queries"][name]["known_dates"] == int(np.isfinite(years.loc[query_ids]).sum()), "summary known-date count mismatch")

    comparison_paths = {
        "grid_T1": args.grid_t1 or data / "results/preds/zhang7_T1_dual_time_available_s0_20261118.parquet",
        "grid_T2": args.grid_t2 or data / "results/preds/zhang7_T2_direct_pretrained_s0_20261118.parquet",
    }
    for name, path in comparison_paths.items():
        full[name], metrics[name] = describe(pd.read_parquet(path), expected_ids, labels, metadata)
    t3_path = args.t3 or run.parent / "2026-11-18-geoclip-t0-photo-rerank/results/predictions.parquet"
    t3 = pd.read_parquet(t3_path)
    require(set(t3.method) == {"visual", "year"}, "unexpected T3 methods")
    for method, group in t3.groupby("method"):
        full["T3_" + method], metrics["T3_" + method] = describe(group, expected_ids, labels, metadata)
    valid_dates = years.notna()
    changes = paired(first, second, valid_dates)
    deltas = {baseline: {name: {key: metrics["year"][name][key] - metrics[baseline][name][key]
                              for key in ("median_km", "mean_km", "acc@25km")}
                        for name in SPLITS} for baseline in ("location", "grid_T1", "grid_T2", "T3_visual", "T3_year")}
    spreads = summary.get("candidate_spread", {})
    require(set(spreads) == set(SPLITS), "candidate spread splits missing")
    for name, row in spreads.items():
        require(row["n"] == counts[name], "candidate spread count mismatch")
        require(np.isfinite(row["candidate_radius_median_km"]) and row["candidate_radius_median_km"] >= 0, "invalid candidate radius")
        require(0 <= row["candidate_radius_le5km_percent"] <= 100, "invalid candidate radius percentage")
    output = {"verified": True, "selected_gamma": gamma, "hyperparameters_changed": False,
              "cached_training_references": len(train_ids), "counts": counts, "same_eval_ids_and_splits": True,
              "metrics": metrics, "paired_location_to_year": changes, "year_minus_baseline": deltas,
              "candidate_spread_reported": spreads,
              "verification_limits": ["Neural cosine values and complete top-K membership were not recomputed.",
                                      "Candidate spread was range/count checked only; full candidates were not exported."],
              "comparison_files": {**{name: str(path) for name, path in comparison_paths.items()}, "T3": str(t3_path)}}
    destination = run / "verified_comparison.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    main()
