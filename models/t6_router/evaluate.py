"""Evaluate the frozen T6 selection model and both of its experts on held-out photos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from models.clip_regression.scripts.compare_t1 import (
    ERROR_ATOL_KM, _errors, _gps, _provenance, _validate_ids,
)
from models.t5_reranker.evaluate import GPS, PREDICTED, original_rows
from models.t6_router.bundle import DEFAULT_BUNDLE, LEGACY_RULE, RULE, load_bundle
from models.t6_router.router import select_expert


ROOT = Path(__file__).resolve().parents[2]
TEST_SPLITS = ("test_in", "test_city", "test_region")
RADII = (1, 5, 25, 50)


def paired_predictions(path, truth, *, models=("T4", "T5")):
    """Require exactly the canonical test queries from each independent calibrator."""
    original = truth.loc[truth.split.isin(TEST_SPLITS)]
    if set(original.block) & set(truth.loc[truth.split.eq("val"), "block"]):
        raise ValueError("A held-out block also occurs in validation")
    if original[["split", "block"]].isna().any().any():
        raise ValueError("Missing canonical test split or block")
    paired = original[["split", "block"]].copy()
    for model in models:
        raw = pd.read_parquet(path, filters=[("model", "=", model), ("split", "in", TEST_SPLITS)])
        _validate_ids(raw, model)
        frame = raw.set_index("id").sort_index()
        if not frame.index.equals(original.index) or not frame.split.eq(original.split).all():
            raise ValueError(f"{model} query IDs or split assignments disagree with the original data")
        if not frame.block.eq(original.block).all() or not frame.confidence_partition.eq("test").all():
            raise ValueError(f"{model} has the wrong test partition or geographic block")
        pred = _gps(frame, PREDICTED, model)
        distances = _errors(original[list(GPS)].to_numpy(), pred)
        if not np.allclose(distances, frame.error_km, rtol=0, atol=ERROR_ATOL_KM):
            raise ValueError(f"{model} saved errors disagree with the canonical GPS")
        prob = frame.p_within_25km.to_numpy(dtype=float)
        if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
            raise ValueError(f"Invalid probabilities for {model}")
        paired[f"{model}_lat_pred"] = pred[:, 0]
        paired[f"{model}_lon_pred"] = pred[:, 1]
        paired[f"{model}_p25"] = prob
        paired[f"{model}_error_km"] = distances
    return paired


def distance_metrics(errors):
    result = {"median_km": float(np.median(errors)), "mean_km": float(np.mean(errors)),
              "p90_km": float(np.quantile(errors, 0.9))}
    for radius in RADII:
        count = int(np.count_nonzero(errors <= radius))
        result[f"hits_{radius}km"] = count
        result[f"hit_{radius}km_pct"] = 100 * count / len(errors)
    return result


def block_interval(blocks, delta):
    """Paired bootstrap of per-photo hit-rate change, resampling entire blocks."""
    block, index = np.unique(blocks, return_inverse=True)
    counts = np.bincount(index)
    gains = np.bincount(index, weights=delta)
    sampled = np.random.default_rng(42).integers(len(block), size=(3000, len(block)))
    pp = 100 * gains[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    lower, upper = np.quantile(pp, [0.025, 0.975])
    return float(lower), float(upper)


def dispatch_metrics(frame):
    use4 = frame.chosen_model.eq("T4").to_numpy()
    err4 = frame.T4_error_km.to_numpy()
    err5 = frame.T5_error_km.to_numpy()
    h4, h5 = err4 <= 25, err5 <= 25
    only4 = h4 & ~h5
    recovered = use4 & only4
    lost = use4 & ~h4 & h5
    # Oracle opportunity and chosen counts are measured with labels only in evaluation.
    delta = recovered.astype(int) - lost.astype(int)
    lower, upper = block_interval(frame.block.to_numpy(), delta)
    return {
        "choose_T4": int(use4.sum()), "choose_T4_pct": float(100 * use4.mean()),
        "equal_scores": int(frame.T4_p25.eq(frame.T5_p25).sum()),
        "rescued_hits_25km": int(recovered.sum()), "lost_hits_25km": int(lost.sum()),
        "net_hits_25km": int(delta.sum()), "delta_25km_vs_T5_pp": 100 * float(delta.mean()),
        "block_ci95_low_pp": lower, "block_ci95_high_pp": upper,
        "total_T4_only_hits_25km": int(only4.sum()),
        "median_T4_p25_when_only_T4_hits": float(frame.T4_p25[only4].median()) if only4.any() else None,
        "median_T5_p25_when_only_T4_hits": float(frame.T5_p25[only4].median()) if only4.any() else None,
        "total_T5_only_hits_25km": int((h5 & ~h4).sum()),
        "both_hit_25km": int((h4 & h5).sum()), "neither_hit_25km": int((~h4 & ~h5).sum()),
        "when_T4_chosen_T4_closer": int((use4 & (err4 < err5)).sum()),
        "when_T4_chosen_T5_closer": int((use4 & (err5 < err4)).sum()),
        "when_T4_chosen_both_hit_25km": int((use4 & h4 & h5).sum()),
        "when_T4_chosen_neither_hit_25km": int((use4 & ~h4 & ~h5).sum()),
    }


def run(path, data_root, split_csv, output_dir, *, model_path=DEFAULT_BUNDLE, legacy_confidence=False):
    if output_dir.exists():
        raise FileExistsError(f"Choose a new T6 evaluation directory: {output_dir}")
    gate = None if legacy_confidence else load_bundle(model_path).get("gate")
    truth, _, sources = original_rows(data_root, split_csv)
    paired = paired_predictions(path, truth)
    use4, probability = select_expert(paired, gate, legacy_confidence=legacy_confidence)
    if probability is not None:
        paired["gate_p_T4_given_one_hit"] = probability
    paired["chosen_model"] = np.where(use4, "T4", "T5")
    paired["selected_expert_p25"] = np.where(use4, paired.T4_p25, paired.T5_p25)
    for name in (*PREDICTED, "error_km"):
        paired[name] = np.where(use4, paired[f"T4_{name}"], paired[f"T5_{name}"])
    # Verify the published T6 coordinates and reported distances as a pair.
    published = _errors(truth.loc[paired.index, list(GPS)].to_numpy(), paired[list(PREDICTED)].to_numpy())
    np.testing.assert_allclose(published, paired.error_km, rtol=0, atol=ERROR_ATOL_KM)

    metrics, routing = [], []
    for split in TEST_SPLITS:
        frame = paired.loc[paired.split.eq(split)]
        key = {"split": split, "n": len(frame), "n_blocks": int(frame.block.nunique())}
        if key["n_blocks"] < 2:
            raise ValueError(f"Insufficient blocks to evaluate {split}")
        oracle = np.minimum(frame.T4_error_km, frame.T5_error_km)
        for name, errors in (("T4", frame.T4_error_km), ("T5", frame.T5_error_km),
                             ("T6", frame.error_km), ("oracle", oracle)):
            metrics.append({**key, "model": name, **distance_metrics(errors)})
        routing.append({**key, **dispatch_metrics(frame)})

    inputs = {**sources, "predictions": path}
    if gate is not None:
        inputs["gate"] = gate
    report = {
        "model_version": 1 if legacy_confidence else 2,
        "rule": LEGACY_RULE if legacy_confidence else RULE,
        "selected_expert_p25": "The selected expert's own probability of a 25 km hit before routing.",
        "oracle": "Evaluation-only upper bound selecting the smaller of the two true errors separately for each photo.",
        "tests": "Official held-out queries only. No thresholds were fitted. Base predictors and per-expert calibrators are frozen.",
        "block_ci": "Percentile 95% interval for paired T6 minus T5 hit-rate change in percentage points; 3,000 draws of spatial blocks with replacement (seed 42). Small block counts limit interpretation.",
        "inputs": {key: _provenance(p) for key, p in inputs.items()},
        "metrics": metrics, "routing": routing,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as temp:
        staging = Path(temp) / "evaluation"
        staging.mkdir()
        paired.reset_index().to_parquet(staging / "predictions.parquet", index=False)
        pd.DataFrame(metrics).to_csv(staging / "metrics.csv", index=False)
        pd.DataFrame(routing).to_csv(staging / "routing.csv", index=False)
        (staging / "comparison.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--predictions", type=Path, default=ROOT / "data/local_runs/confidence_models/predictions.parquet")
    cli.add_argument("--data-root", type=Path, default=ROOT / "data/t1_original")
    cli.add_argument("--split-csv", type=Path, default=ROOT / "data/splits/main.csv.gz")
    cli.add_argument("--model", type=Path, default=DEFAULT_BUNDLE, help="T6 v2 manifest")
    cli.add_argument("--legacy-confidence", action="store_true", help="Explicitly reproduce the v1 confidence switch")
    cli.add_argument("--output-dir", type=Path, help="New directory for evaluated predictions and metrics")
    args = cli.parse_args()
    output = args.output_dir or DEFAULT_BUNDLE.parent.parent / (
        "evaluation_v1_replay" if args.legacy_confidence else "evaluation_v2"
    )
    outcome = run(args.predictions, args.data_root, args.split_csv, output,
                  model_path=args.model, legacy_confidence=args.legacy_confidence)
    print(pd.DataFrame(outcome["routing"])[[
        "split", "n", "choose_T4", "rescued_hits_25km", "lost_hits_25km",
        "delta_25km_vs_T5_pp", "block_ci95_low_pp", "block_ci95_high_pp",
    ]].to_string(index=False))
