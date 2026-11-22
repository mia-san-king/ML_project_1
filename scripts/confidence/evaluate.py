"""Fit one 25 km confidence calibrator per frozen geolocator and score its test predictions."""
import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.confidence import GeoConfidence


DEFAULT_RUN = ROOT / "data/local_runs/confidence_models"
PREDICTION_CONTEXT = ["nearest_train_km", "log1p_train_5km", "log1p_train_25km", "has_date"]
FEATURES = {
    "T1": PREDICTION_CONTEXT,
    "T3": ["t3_visual_score", "t3_year_penalty", "has_date"],
    "T4": ["t4_was_projected", *PREDICTION_CONTEXT],
    "T5": ["t5_visual_rank", "t5_t3_score", "t5_location_cosine"],
}
EXPECTED = {"val": 29529, "test_in": 47593, "test_city": 3377, "test_region": 13039}
TEST_SPLITS = ("test_in", "test_city", "test_region")
RADIUS_KM = 25.0
SEED = 42


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(table, split_path, clip_ids_path):
    if set(table.model) != set(FEATURES) or table.duplicated(["model", "id"]).any():
        raise ValueError("Expected exactly one row for each model and original query")
    if table[["id", "split", "block"]].isna().any().any():
        raise ValueError("Missing query metadata")
    numeric = table[["lat_pred", "lon_pred", "error_km"]]
    if not np.isfinite(numeric).all().all() or (table.error_km < 0).any():
        raise ValueError("Nonfinite coordinates or invalid error")
    if (table.lat_pred.abs() > 90).any() or (table.lon_pred.abs() > 180).any():
        raise ValueError("Invalid prediction coordinates")

    # Check query IDs and group membership against the original split independently of the models.
    all_queries = pd.read_csv(split_path, usecols=["id", "split", "block"])
    clip_ids = np.load(clip_ids_path)
    available = all_queries.split.isin(EXPECTED) & all_queries.id.isin(clip_ids)
    original = all_queries.loc[available].set_index("id", verify_integrity=True).sort_index()
    if original.split.value_counts().to_dict() != EXPECTED:
        raise ValueError("Unexpected original query coverage")
    for model, bundle in table.groupby("model"):
        per_query = bundle.set_index("id").sort_index()
        if not per_query.index.equals(original.index) or not original.split.eq(per_query.split).all():
            raise ValueError(f"Incorrect IDs or splits for {model}")
        if not original.block.astype(str).eq(per_query.block.astype(str)).all():
            raise ValueError(f"Incorrect spatial block for {model}")
        values = bundle[FEATURES[model]].to_numpy(dtype=float)
        if np.isinf(values).any() or np.isnan(values).all(axis=0).any():
            raise ValueError(f"Invalid deployment feature for {model}")
    blocks = {split: set(original.loc[original.split.eq(split), "block"])
              for split in EXPECTED}
    overlaps = {f"{a}/{b}": len(blocks[a] & blocks[b]) for a, b in combinations(EXPECTED, 2)}
    if any(blocks["val"] & blocks[split] for split in TEST_SPLITS):
        raise ValueError("Spatial overlap between validation and final testing")
    return overlaps


def score_bins(labels, probabilities):
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid output probability")
    bin_id = np.minimum((probabilities * 10).astype(int), 9)
    bins = []
    weighted_gap = 0.0
    for idx in range(10):
        mask = bin_id == idx
        count = int(mask.sum())
        pred = float(probabilities[mask].mean()) if count else None
        actual = float(labels[mask].mean()) if count else None
        if count:
            weighted_gap += count * abs(pred - actual)
        bins.append(dict(lower=idx / 10, upper=(idx + 1) / 10, n=count,
                         mean_probability=pred, hit_rate=actual))
    summary = {
        "n": len(labels), "hits": int(labels.sum()),
        "mean_probability": float(probabilities.mean()), "hit_rate": float(labels.mean()),
        "ece_10": weighted_gap / len(labels), "brier": float(np.square(probabilities - labels).mean()),
        "auroc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size > 1 else None,
    }
    return summary, bins


def run(input_path, out_dir, split_path, clip_ids_path):
    table = pd.read_parquet(input_path)
    overlaps = verify_inputs(table, split_path, clip_ids_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics, predictions, reliability_bins, artifacts = [], [], [], {}
    expected_assignment = None
    for model, features in FEATURES.items():
        bundle = table.loc[table.model.eq(model)].sort_values("id").reset_index(drop=True)
        val = bundle.loc[bundle.split.eq("val")]
        groups = val.block.astype(str)
        cal = GeoConfidence(features, threshold_km=RADIUS_KM, random_state=SEED)
        cal.fit(val[features], val.error_km, groups=groups, query_ids=val.id)
        fit_blocks = set(map(str, cal.fit_groups_))
        calibration_blocks = set(map(str, cal.calibration_groups_))
        if fit_blocks & calibration_blocks or fit_blocks | calibration_blocks != set(groups):
            raise RuntimeError("Validation block assignment is incomplete or overlapping")
        assignment = (tuple(sorted(fit_blocks)), tuple(sorted(calibration_blocks)))
        if expected_assignment is not None and expected_assignment != assignment:
            raise RuntimeError("Different models used different block partitions")
        expected_assignment = assignment
        fit = val.loc[groups.isin(fit_blocks)]
        baseline = float(fit.error_km.le(RADIUS_KM).mean())
        artifact = out_dir / f"{model}.joblib"
        cal.save(artifact)
        probabilities = cal.predict_proba(bundle[features])
        loaded = GeoConfidence.load(artifact)
        np.testing.assert_array_equal(loaded.predict_proba(bundle[features]), probabilities)
        if probabilities.shape != (len(bundle),):
            raise RuntimeError("Expected one probability per query")

        test = bundle.split.isin(TEST_SPLITS)
        stages = np.where(test, "test", np.where(bundle.block.astype(str).isin(fit_blocks),
                                                 "fit", "calibration"))
        pred = bundle[["id", "model", "split", "block", "lat_pred", "lon_pred", "error_km"]].copy()
        pred["confidence_partition"] = stages
        pred["p_within_25km"] = probabilities
        predictions.append(pred)
        artifacts[model] = {
            "file": artifact.name, "sha256": sha256(artifact), "features": features,
            "n_fit": len(fit), "n_calibration": len(val) - len(fit),
            "baseline_fit_hit_rate": baseline, "reload_exact": True,
        }
        for split in TEST_SPLITS:
            mask = bundle.split.eq(split).to_numpy()
            labels = bundle.loc[mask, "error_km"].le(RADIUS_KM).to_numpy(dtype=int)
            for method, p in (("per_model", probabilities[mask]), ("constant", np.full(len(labels), baseline))):
                summary, bins = score_bins(labels, p)
                key = {"model": model, "split": split, "method": method}
                metrics.append({**key, "n_blocks": int(bundle.loc[mask, "block"].nunique()), **summary})
                reliability_bins.append({**key, "bins": bins})
        print(f"{model}: saved and verified {artifact.name}; fit={len(fit)}, calibration={len(val) - len(fit)}",
              flush=True)

    predictions = pd.concat(predictions, ignore_index=True)
    prediction_file = out_dir / "predictions.parquet"
    predictions.to_parquet(prediction_file, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(prediction_file), predictions)
    pd.DataFrame(metrics).to_csv(out_dir / "metrics.csv", index=False)
    report = {
        "event": "Great-circle prediction error <= 25 km",
        "estimators": "One confidence estimator per frozen geolocator with 8,192 diffusion training images.",
        "splits": {"val": "Confidence fit/calibration only", "test_in": "In-country held-out blocks",
                   "test_city": "Held-out city of Bern", "test_region": "Held-out region of Ticino"},
        "fit_procedure": "Seed 42. GroupShuffleSplit on the sorted unique validation block IDs; 30% of blocks held for calibration. Identical partition for every model.",
        "fit_blocks": expected_assignment[0], "calibration_blocks": expected_assignment[1],
        "baseline": "Model-specific hit rate on the estimator-fitting portion of validation, assigned to all queries.",
        "ece": "Sample-weighted absolute predicted vs observed hit-rate gaps, using 10 fixed probability bins. Intervals are left-closed, right-open, except [0.9, 1.0].",
        "limitations": [
            "The geolocators were frozen before this confidence experiment. Their original development used validation data, so only the three test splits evaluate the confidence layer.",
            "The held-out city and region represent geographic shifts. Calibration fitted on validation blocks does not guarantee matching hit rates in these areas.",
            "Metrics weight each query equally; photographs in the same block may be correlated. Rows for different models refer to the same query IDs.",
        ],
        "input": {"file": input_path.name, "sha256": sha256(input_path)},
        "reference_files": {path.name: sha256(path) for path in (split_path, clip_ids_path)},
        "block_overlap_counts": overlaps, "unique_queries_per_split": EXPECTED,
        "sklearn_version": sklearn.__version__, "artifacts": artifacts,
        "metrics": metrics, "reliability_bins": reliability_bins,
        "predictions": {"file": prediction_file.name, "rows": len(predictions),
                        "sha256": sha256(prediction_file)},
    }
    (out_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RUN / "inputs.parquet")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--split-csv", type=Path, default=ROOT / "data/splits/main.csv.gz")
    parser.add_argument("--clip-ids", type=Path,
                        default=ROOT / "data/t1_original/features/clip_vitl14_ids.npy")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.input, args.out_dir, args.split_csv, args.clip_ids)
