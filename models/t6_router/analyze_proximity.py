"""Audit T4/T5 by true gallery distance and by inference-time T5 confidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.confidence_features import GalleryFeatureExtractor
from models.clip_regression.scripts.compare_t1 import ERROR_ATOL_KM, _errors
from models.t5_reranker.evaluate import GPS, PREDICTED, original_rows


ROOT = Path(__file__).resolve().parents[2]
TEST_SPLITS = ("test_in", "test_city", "test_region")


def summarize(group):
    row = {"n": len(group)}
    for model in ("T4", "T5"):
        errors = group[f"{model}_error_km"]
        row[model] = {
            "hits_25km": int(errors.le(25).sum()),
            "hit_rate_25km": float(errors.le(25).mean()) if len(group) else None,
            "median_km": float(errors.median()) if len(group) else None,
        }
    row["mean_t5_confidence"] = float(group.p5.mean()) if len(group) else None
    row["median_true_gallery_distance_km"] = float(group.true_gallery_km.median()) if len(group) else None
    return row


def analyze(data_root, splits, predictions):
    truth, gallery, _ = original_rows(data_root, splits)
    table = pd.read_parquet(predictions, filters=[("model", "in", ["T4", "T5"])])
    cases = truth[["split"]].copy()
    for model in ("T4", "T5"):
        per_model = table.loc[table.model.eq(model)].set_index("id", verify_integrity=True).sort_index()
        if not per_model.index.equals(truth.index) or not per_model.split.equals(truth.split):
            raise ValueError(f"Canonical query IDs or splits differ for {model}")
        errors = _errors(truth[list(GPS)].to_numpy(), per_model[list(PREDICTED)].to_numpy())
        if not np.allclose(errors, per_model.error_km, rtol=0, atol=ERROR_ATOL_KM):
            raise ValueError(f"Stored {model} errors disagree with coordinate distances")
        cases[f"{model}_error_km"] = errors
        if model == "T5":
            gallery_gps = pd.MultiIndex.from_frame(gallery[list(GPS)])
            if not pd.MultiIndex.from_frame(per_model[list(PREDICTED)]).isin(gallery_gps).all():
                raise ValueError("Expected a real gallery coordinate for every T5 retrieval")
            cases["p5"] = per_model.p_within_25km
            if not cases.p5.between(0, 1).all():
                raise ValueError("Missing or invalid T5 probabilities")
            reference = per_model.split.eq("val") & per_model.confidence_partition.eq("calibration")

    # Ground truth enters this diagnostic only. These distances are unavailable during routing.
    geo = GalleryFeatureExtractor(gallery[list(GPS)].to_numpy())
    cases["true_gallery_km"] = geo.transform(
        truth[list(GPS)].to_numpy(), np.zeros(len(truth), dtype=bool), workers=4,
    ).nearest_train_km.to_numpy()
    if (cases.true_gallery_km > cases.T5_error_km + ERROR_ATOL_KM).any():
        raise ValueError("A T5 prediction is closer than the nearest gallery coordinate")

    if not reference.any():
        raise ValueError("Missing validation-calibration queries for fixed probability cutoffs")
    low, high = cases.loc[reference, "p5"].quantile([0.25, 0.75]).tolist()
    if low >= high:
        raise ValueError("Degenerate validation probability quartiles")
    categories = {
        "true_gallery_distance": {
            "0_to_5km": cases.true_gallery_km.le(5),
            "over_5_to_25km": cases.true_gallery_km.gt(5) & cases.true_gallery_km.le(25),
            "over_25km": cases.true_gallery_km.gt(25),
        },
        "deployable_t5_confidence": {
            "lower_quartile": cases.p5.le(low),
            "middle_half": cases.p5.gt(low) & cases.p5.lt(high),
            "upper_quartile": cases.p5.ge(high),
        },
    }
    results = {}
    for split in TEST_SPLITS:
        scope = cases.split.eq(split)
        results[split] = {"all": summarize(cases.loc[scope])}
        for name, masks in categories.items():
            results[split][name] = {label: summarize(cases.loc[scope & mask]) for label, mask in masks.items()}
            if sum(item["n"] for item in results[split][name].values()) != int(scope.sum()):
                raise ValueError(f"{name} bins do not partition all {split} queries")
    return {
        "gallery_size": len(gallery),
        "distance_definition": "Great-circle distance from true test GPS to nearest real training photo; evaluation only.",
        "p5_quartiles": {"reference": "val calibration blocks, without hit labels",
                         "n": int(reference.sum()), "q25": low, "q75": high},
        "splits": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/t1_original")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/splits/main.csv.gz")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/local_runs/confidence_models/predictions.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/local_runs/t6_aug8192/proximity.json")
    args = parser.parse_args()
    report = analyze(args.data_root, args.splits, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
