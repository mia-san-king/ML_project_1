"""Fit a regional T6 selector to choose the closer frozen expert on each photo."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from models.t6_router.geographic_gate import GeographicGate, NONPOSITION, PARAMETERS, POSITION, _errors, _features
from models.t6_router.region_gate import (
    CURRENT, REGIONAL, SAME_CONF, _validate_partition, confidence_scores, fit_confidence, paired_change,
)
from scripts.confidence.evaluate import RADIUS_KM, sha256


CLOSER = "regional_closer"
FEATURES = POSITION + NONPOSITION


class CloserGate(GeographicGate):
    """Probability that T4 is closer, conditional on unequal expert distances."""


def fit_closer(frame):
    x = _features(frame, FEATURES)
    e4, e5 = _errors(frame).T
    unequal = e4 != e5
    y = (e4[unequal] < e5[unequal]).astype(int)
    if len(np.unique(y)) < 2:
        return CloserGate(FEATURES, None, float(y[0]) if len(y) else 0.5)
    with threadpool_limits(limits=4):
        estimator = HistGradientBoostingClassifier(**PARAMETERS).fit(x[unequal], y)
    return CloserGate(FEATURES, estimator, .5)


def _choices(frame, picks):
    array = np.asarray(picks)
    if array.shape != (len(frame),) or array.dtype.kind != "b":
        raise ValueError("Expected one boolean choice per photo")
    return array


def distance_metrics(frame, picks):
    picks = _choices(frame, picks)
    e4, e5 = _errors(frame).T
    error = np.where(picks, e4, e5)
    oracle = np.minimum(e4, e5)
    unequal = e4 != e5
    n = int(unequal.sum())
    correct = int(((error == oracle) & unequal).sum())
    return {"n": len(frame), "n_non_ties": n, "n_ties": len(frame) - n, "correct_non_ties": correct,
            "accuracy_non_ties_pct": 100. * correct / n if n else None, "choose_T4": int(picks.sum()),
            "mean_km": float(error.mean()), "median_km": float(np.median(error)),
            "mean_oracle_regret_km": float((error - oracle).mean()), "hits_25km": int((error <= RADIUS_KM).sum()),
            "hit_25km_pct": float(100 * (error <= RADIUS_KM).mean())}


def paired_distances(frame, new, old):
    new, old = _choices(frame, new), _choices(frame, old)
    e4, e5 = _errors(frame).T
    previous, candidate = np.where(old, e4, e5), np.where(new, e4, e5)
    switched = old != new
    difference = previous - candidate
    better, worse = difference > 0, difference < 0
    saved, added = float(difference[better].sum()), float(-difference[worse].sum())
    net = saved - added
    np.testing.assert_allclose(net, previous.sum() - candidate.sum(), rtol=1e-10, atol=1e-7)
    return {"switches": int(switched.sum()), "T5_to_T4": int((~old & new).sum()),
            "T4_to_T5": int((old & ~new).sum()), "improved": int(better.sum()), "worsened": int(worse.sum()),
            "T4_closer_newly_correct": int(((e4 < e5) & better).sum()),
            "T4_closer_previously_correct_lost": int(((e4 < e5) & worse).sum()),
            "T5_closer_newly_correct": int(((e5 < e4) & better).sum()),
            "T5_closer_previously_correct_lost": int(((e5 < e4) & worse).sum()),
            "equal_after_switch": int((switched & (difference == 0)).sum()),
            "saved_km": saved, "added_km": added, "net_saved_km": net, "mean_saved_km": net / len(frame),
            "net_correct_non_ties": int(better.sum()) - int(worse.sum())}


def _development_inputs(development):
    rows = development.reset_index(drop=True).copy()
    probabilities = np.full((len(rows), 2), np.nan)
    folds = {}
    for name in sorted(rows.region.unique()):
        hold = rows.region.eq(name).to_numpy()
        probabilities[hold] = confidence_scores(rows.loc[hold], fit_confidence(rows.loc[~hold]))
        e4, e5 = _errors(rows.loc[hold]).T
        folds[str(name)] = {"n": len(e4), "T4_closer": int((e4 < e5).sum()), "T5_closer": int((e5 < e4).sum()),
                            "exact_ties": int((e4 == e5).sum())}
    rows[list(NONPOSITION)] = probabilities
    return rows, folds


def _archived(final, reference_dir):
    path = reference_dir / "decisions.parquet"
    decisions = pd.read_parquet(path).set_index("id", verify_integrity=True)
    if set(decisions.index) != set(final.id):
        raise ValueError("Archived regional routing decisions differ from the reserved query IDs")
    decisions = decisions.loc[final.id].reset_index()
    for field in ("region", "block"):
        if not np.array_equal(decisions[field].astype(str), final[field].astype(str)):
            raise ValueError(f"Archived {field} differs from the reserved queries")
    np.testing.assert_array_equal(decisions[list(POSITION)].to_numpy(), final[list(POSITION)].to_numpy())
    for policy in (CURRENT, SAME_CONF, REGIONAL):
        p = decisions[f"{policy}_p_T4"].to_numpy()
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid historical gate score")
        np.testing.assert_array_equal(_choices(final, decisions[f"{policy}_choose_T4"]), p > .5)
    return decisions


def run_closer(development, final, *, reference_dir, output_dir):
    """Compare the new objective on exactly the regional study's final expert outputs."""
    reference, output = Path(reference_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    _validate_partition(development, final)
    final = final.reset_index(drop=True)
    frozen_path = reference / "regional_gate.joblib"
    with threadpool_limits(limits=4):
        training, folds = _development_inputs(development)
        gate = fit_closer(training)
        decisions = _archived(final, reference)
        frozen = joblib.load(frozen_path)
        confidence = confidence_scores(final, frozen["confidence"])
        recorded = decisions[[f"regional_{name}" for name in NONPOSITION]].to_numpy()
        np.testing.assert_allclose(confidence, recorded, rtol=0, atol=1e-12)
        inputs = final.loc[:, list(POSITION)].copy()
        inputs[list(NONPOSITION)] = recorded
        np.testing.assert_allclose(frozen["gate"].predict_t4_probability(inputs), decisions[f"{REGIONAL}_p_T4"],
                                   rtol=0, atol=1e-12)
        probability = gate.predict_t4_probability(inputs)
    decisions[f"{CLOSER}_p_T4"] = probability
    decisions[f"{CLOSER}_choose_T4"] = probability > .5
    err = _errors(final)
    np.testing.assert_array_equal(err, _errors(decisions))
    e4, e5 = err.T
    decisions["oracle_error_km"] = np.minimum(e4, e5)
    methods = {}
    for policy in (CURRENT, SAME_CONF, REGIONAL, CLOSER):
        picks = decisions[f"{policy}_choose_T4"]
        decisions[f"{policy}_error_km"] = np.where(picks, e4, e5)
        methods[policy] = distance_metrics(decisions, picks)
    methods["always_T4"] = distance_metrics(decisions, np.ones(len(final), dtype=bool))
    methods["always_T5"] = distance_metrics(decisions, np.zeros(len(final), dtype=bool))
    comparisons = {}
    for baseline in (CURRENT, REGIONAL):
        result = paired_distances(decisions, decisions[f"{CLOSER}_choose_T4"], decisions[f"{baseline}_choose_T4"])
        result["hit25_changes"] = paired_change(decisions, decisions[f"{CLOSER}_choose_T4"].to_numpy(),
                                                 decisions[f"{baseline}_choose_T4"].to_numpy())
        if result["hit25_changes"]["net_hits_25km"] != methods[CLOSER]["hits_25km"] - methods[baseline]["hits_25km"]:
            raise RuntimeError("Exclusive 25 km hit transitions disagree with observed hit totals")
        if result["net_correct_non_ties"] != methods[CLOSER]["correct_non_ties"] - methods[baseline]["correct_non_ties"]:
            raise RuntimeError("Switch counts disagree with the difference in correct choices")
        np.testing.assert_allclose(result["mean_saved_km"], methods[baseline]["mean_km"] - methods[CLOSER]["mean_km"],
                                   rtol=1e-10, atol=1e-9)
        comparisons[f"vs_{baseline}"] = result
    report = {
        "n": len(final), "final_region": str(final.region.iloc[0]), "development_n": len(training),
        "development_regions": folds, "T4_closer": int((e4 < e5).sum()), "T5_closer": int((e5 < e4).sum()),
        "exact_ties": int((e4 == e5).sum()), "oracle_mean_km": float(decisions.oracle_error_km.mean()),
        "description": "Unweighted classification on all development photos with unequal expert distances. "
                       "Strictly greater than 0.5 selects T4. Exact equal distances are excluded from accuracy "
                       "and included in all distance and 25 km metrics. Positive saved km means lower error.",
        "inputs": {"prior_decisions_sha256": sha256(reference / "decisions.parquet"),
                   "regional_confidence_and_prior_gate_sha256": sha256(frozen_path)},
        "methods": methods, "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "closer"
        stage.mkdir()
        joblib.dump({"gate": gate, "confidence": frozen["confidence"]}, stage / "closer_gate.joblib", compress=3)
        decisions.to_parquet(stage / "decisions.parquet", index=False)
        report["decisions_sha256"] = sha256(stage / "decisions.parquet")
        report["artifact_sha256"] = sha256(stage / "closer_gate.joblib")
        (stage / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    return report
