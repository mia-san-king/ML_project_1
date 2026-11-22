"""Compare local T3/T4 predictions against the verified original evaluation photos."""
import hashlib
import json
import sys
from pathlib import Path
from experiment_paths import T1_ORIGINAL, T1_AUGMENTED

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "data/local_runs/t3_t4_8192"
DATA = ROOT / "data/t1_original"
EXPERIMENTS = ROOT / "report/experiments"
REMOTE = EXPERIMENTS / "2026-11-20-diffusion-augmentation"
EXPECTED = {"val": 29529, "test_in": 47593, "test_city": 3377, "test_region": 13039}
sys.path.insert(0, str(REMOTE))
from compare_augmented import error_km, metrics, read_predictions
from paired_blocks import MODELS, bootstrap


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "sha256": digest.hexdigest()}


def authoritative_data():
    archive = EXPERIMENTS / "2026-11-18-geoclip-time/run/input_manifest.json"
    hashed = {Path(row["path"]).name: row["sha256"] for row in read_json(archive)["inputs"]}
    split_path = ROOT / "data/splits/main.csv.gz"
    gps_path = DATA / "meta/smapshot.parquet"
    ids_path = DATA / "features/clip_vitl14_ids.npy"
    inputs = {}
    for path in (split_path, gps_path, ids_path):
        inputs[path.name] = source(path)
        if inputs[path.name]["sha256"] != hashed[path.name]:
            raise ValueError(f"Original data checksum mismatch: {path}")
    split = pd.read_csv(split_path, usecols=["id", "split", "block"]).set_index("id", verify_integrity=True)
    gps = pd.read_parquet(gps_path, columns=["id", "latitude", "longitude"]).set_index("id", verify_integrity=True)
    ids = pd.Index(np.load(ids_path))
    if not ids.is_unique or (ids <= 0).any():
        raise ValueError("Original feature cache has invalid IDs")
    available = split.loc[split.index.isin(ids)].join(gps, validate="one_to_one")
    truth = available.loc[available.split.isin(EXPECTED)].sort_index()
    if truth.split.value_counts().to_dict() != EXPECTED or not np.isfinite(truth[["latitude", "longitude"]]).all().all():
        raise ValueError("Authoritative evaluation coverage or GPS failed")
    return truth, available.loc[available.split.eq("train")], inputs


def checked_errors(path, model, truth, train):
    predictions = read_predictions(path, model)
    if not predictions.index.equals(truth.index) or not predictions.split.equals(truth.split):
        raise ValueError(f"Prediction IDs/splits differ from verified original data: {path}")
    if model == "T4":
        np.testing.assert_array_equal(predictions[["latitude", "longitude"]], truth[["latitude", "longitude"]])
    if model == "T3":
        if not predictions.neighbor_split.eq("train").all():
            raise ValueError(f"Reference split differs from train: {path}")
        reference_truth = train.loc[predictions.neighbor_id, ["latitude", "longitude"]]
        np.testing.assert_array_equal(predictions[["lat_pred", "lon_pred"]], reference_truth)
    errors = error_km(predictions, truth)
    if not np.isfinite(errors).all():
        raise ValueError(f"Nonfinite distance: {path}")
    if "error_km" in predictions:
        np.testing.assert_allclose(errors, predictions.error_km, rtol=1e-9, atol=1e-7)
    return errors


def main():
    truth, train, inputs = authoritative_data()
    keep_test = truth.split.eq("test_in").to_numpy()
    test_blocks = truth.loc[keep_test, "block"]
    if test_blocks.isna().any() or test_blocks.nunique() != 93:
        raise ValueError("Expected 93 full test_in blocks")
    test_pairs, local, provenance = [], {}, {}
    for model in MODELS:
        paths = [RUN / group / model / "predictions.parquet" for group in ("original", "augmented")]
        if model == "T1":
            paths = [T1_ORIGINAL / "results/preds/zhang7_T1_dual_time_available_s0_20261118.parquet",
                     T1_AUGMENTED / "results/preds/T1_aug8192_s0.parquet"]
        errors = [checked_errors(path, model, truth, train) for path in paths]
        provenance[model] = dict(zip(("original", "augmented_8192"), map(source, paths)))
        test_pairs.append(np.column_stack(errors)[keep_test])
        local[model] = {}
        for split, count in EXPECTED.items():
            if model == "T1" and split != "test_in":
                continue
            mask = truth.split.eq(split).to_numpy()
            old, new = (e[mask] for e in errors)
            before, after = metrics(old), metrics(new)
            gained = int(((new <= 25) & (old > 25)).sum())
            lost = int(((old <= 25) & (new > 25)).sum())
            local[model][split] = {
                "n": count, "original": before, "augmented_8192": after,
                "median_reduction_km": before["median_km"] - after["median_km"],
                "acc25_gain_pp": after["pct25"] - before["pct25"],
                "gained_hits": gained, "lost_hits": lost, "net_hits": gained - lost,
            }
    boot = bootstrap(np.stack(test_pairs, axis=1), test_blocks.to_numpy())
    for model, result in boot.pop("models").items():
        paired = local[model]["test_in"]
        paired["median_reduction_km_ci95"] = result["median_reduction_95pct_km"]
        paired["acc25_gain_pp_ci95"] = result["acc25_gain_95pct_percentage_points"]
    old_t1 = read_json(T1_AUGMENTED / "comparison.json")["splits"]["test_in"]
    np.testing.assert_allclose(local["T1"]["test_in"]["median_reduction_km_ci95"], old_t1["median_gain_km_ci95"])
    np.testing.assert_allclose(local["T1"]["test_in"]["acc25_gain_pp_ci95"], old_t1["acc25_gain_pp_ci95"])
    remote_path = REMOTE / "comparison.json"
    remote = {model: {} for model in ("T3", "T4")}
    for row in read_json(remote_path):
        if row["model"] in remote:
            if row["n"] != EXPECTED[row["split"]]:
                raise ValueError("Remote split count mismatch")
            remote[row["model"]][row["split"]] = {
                "original": row["original"], "augmented_512": row["augmented"],
            }
    result = {
        "local": local, "remote_archive": remote, "remote_source": source(remote_path),
        "method": "Haversine km, radius 6371.0088; T3 method=year; pct25 is percent of images at <=25 km.",
        "bootstrap_test_in": boot,
        "verified_original_queries": {"n": len(truth), "splits": EXPECTED,
            "sorted_ids_le64_sha256": hashlib.sha256(truth.index.to_numpy(dtype="<i8").tobytes()).hexdigest(),
            "inputs": inputs},
        "local_prediction_sources": provenance,
    }
    output = RUN / "comparison.json"
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(output)
    for model in MODELS:
        print(model, json.dumps(local[model]["test_in"]), flush=True)


if __name__ == "__main__":
    main()
