"""Fit the six-input T6 using queries from simulated regional holdouts."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lib.confidence import GeoConfidence
from models.t6_router.geographic_gate import NONPOSITION, POSITION, _errors, fit_gate
from scripts.confidence.evaluate import FEATURES, RADIUS_KM, SEED, sha256


EXPERTS = ("T4", "T5")
CURRENT = "current_T6"
SAME_CONF = "current_gate_regional_confidence"
REGIONAL = "regional_T6"


def fit_confidence(frame):
    """Fit branch probabilities using only labels from the supplied development regions."""
    models = {}
    for model in EXPERTS:
        errors = frame[f"{model}_error_km"]
        if frame.block.nunique() == 1:
            # A separate sigmoid calibration block is unavailable in this fold.
            models[model] = float(((errors <= RADIUS_KM).sum() + 1) / (len(frame) + 2))
        else:
            models[model] = GeoConfidence(FEATURES[model], RADIUS_KM, SEED).fit(
                frame, errors, groups=frame.block, query_ids=frame.id)
    return models


def confidence_scores(frame, models):
    columns = []
    for model in EXPERTS:
        calibrator = models[model]
        if isinstance(calibrator, float):
            columns.append(np.full(len(frame), calibrator))
        else:
            columns.append(calibrator.predict_proba(frame))
    return np.column_stack(columns)


def _exclusive(frame):
    t4, t5 = (_errors(frame) <= RADIUS_KM).T
    return t4 & ~t5, t5 & ~t4


def paired_change(frame, new, old):
    """Account for all four possible changes on photos hit by exactly one expert."""
    only4, only5 = _exclusive(frame)
    to4, to5 = new & ~old, old & ~new
    gained4, reversed4 = int((only4 & to4).sum()), int((only4 & to5).sum())
    lost5, restored5 = int((only5 & to4).sum()), int((only5 & to5).sum())
    return {
        "additional_T4_only_rescues": gained4, "previous_T4_only_rescues_reversed": reversed4,
        "additional_T5_only_losses": lost5, "previous_T5_only_losses_recovered": restored5,
        "net_T4_only_rescues": gained4 - reversed4, "net_T5_only_losses": lost5 - restored5,
        "net_hits_25km": gained4 - reversed4 - lost5 + restored5,
    }


def _metrics(frame, picks):
    only4, only5 = _exclusive(frame)
    err = np.where(picks, frame.T4_error_km, frame.T5_error_km)
    return {
        "choose_T4": int(picks.sum()), "hits_25km": int((err <= RADIUS_KM).sum()),
        "hit_25km_pct": float(100 * (err <= RADIUS_KM).mean()), "median_km": float(np.median(err)),
        "T4_only_rescued": int((picks & only4).sum()), "T5_only_lost": int((picks & only5).sum()),
    }


def _validate_partition(development, final):
    for frame in (development, final):
        if (not len(frame) or not frame.columns.is_unique or frame.id.duplicated().any()
                or frame[["id", "region", "block"]].isna().any().any() or (frame.id <= 0).any()):
            raise ValueError("Expected unique real query IDs, regions, and geographic blocks")
    if set(development.id) & set(final.id) or set(development.region) & set(final.region):
        raise ValueError("Final and development query IDs or regions overlap")
    if final.region.nunique() != 1 or development.region.nunique() < 2:
        raise ValueError("Use one final region and at least two development regions")


def _train(development):
    rows = development.copy().reset_index(drop=True)
    _errors(rows)
    probabilities = np.full((len(rows), 2), np.nan)
    folds = {}
    for name in sorted(rows.region.unique()):
        hold = rows.region.eq(name).to_numpy()
        train = rows.loc[~hold]
        probabilities[hold] = confidence_scores(rows.loc[hold], fit_confidence(train))
        only4, only5 = _exclusive(rows.loc[hold])
        folds[str(name)] = {"n": int(hold.sum()), "only_T4_hits": int(only4.sum()),
                            "only_T5_hits": int(only5.sum()), "confidence_training_n": len(train)}
    rows[list(NONPOSITION)] = probabilities
    return fit_gate(rows), fit_confidence(rows), folds


def run_region_gate(development, final, *, bundle_assets, output_dir):
    """Keep both experts fixed for all three routing policies on the final region.

    Upstream folds must already omit the final region (including its buffer) and
    independently exclude each simulated query region from training and retrieval.
    The source region label is used for splitting only. All gates use the published
    six inputs and class threshold. Final outcomes are read only when scoring.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new routing output directory: {output}")
    _validate_partition(development, final)
    historical_gate = joblib.load(bundle_assets["gate"])
    historical_conf = {m: GeoConfidence.load(bundle_assets[m.lower() + "_confidence"]) for m in EXPERTS}
    if any(c.threshold_km != RADIUS_KM or tuple(c.feature_names) != tuple(FEATURES[m])
           for m, c in historical_conf.items()):
        raise ValueError("Historical confidence schema or hit radius changed")
    with threadpool_limits(limits=4):
        gate, calibrators, folds = _train(development)
        inputs = final.loc[:, list(POSITION)].copy()
        decisions = final[["id", "region", "block", *POSITION]].copy()
        for prefix, models in (("original", historical_conf), ("regional", calibrators)):
            probabilities = confidence_scores(final, models)
            inputs[list(NONPOSITION)] = probabilities
            decisions[[f"{prefix}_{c}" for c in NONPOSITION]] = probabilities
            policies = [(CURRENT, historical_gate)] if prefix == "original" else [(SAME_CONF, historical_gate), (REGIONAL, gate)]
            for name, policy in policies:
                p = policy.predict_t4_probability(inputs)
                decisions[f"{name}_p_T4"] = p
                decisions[f"{name}_choose_T4"] = p > .5

    decisions[["T4_error_km", "T5_error_km"]] = final[["T4_error_km", "T5_error_km"]]
    only4, only5 = _exclusive(decisions)
    methods, comparisons = {}, {}
    for name in (CURRENT, SAME_CONF, REGIONAL):
        methods[name] = _metrics(decisions, decisions[f"{name}_choose_T4"].to_numpy())
    for baseline in (CURRENT, SAME_CONF):
        changes = paired_change(decisions, decisions[f"{REGIONAL}_choose_T4"].to_numpy(),
                                decisions[f"{baseline}_choose_T4"].to_numpy())
        if changes["net_hits_25km"] != methods[REGIONAL]["hits_25km"] - methods[baseline]["hits_25km"]:
            raise RuntimeError("Paired transitions do not reproduce the independent total hit difference")
        comparisons[f"vs_{baseline}"] = changes
    report = {
        "final_region": str(final.region.iloc[0]), "n": len(final), "n_blocks": int(final.block.nunique()),
        "development_regions": folds, "development_n": len(development), "radius_km": RADIUS_KM,
        "confidence": "Leave each development region out when producing its confidence inputs. "
                      "Within the remaining data, split fit and sigmoid calibration by geographic block. "
                      "Fit final confidence on all development regions.",
        "current_T6": "Published gate and published calibrators on the common fold-specific expert predictions.",
        "current_gate_regional_confidence": "Published gate on exactly the same six inputs as regional_T6.",
        "regional_T6": "Same six-input model and fixed 0.5 cutoff, fitted on development-region exclusive hits.",
        "only_T4_hits": int(only4.sum()), "only_T5_hits": int(only5.sum()),
        "T4_hits": int((final.T4_error_km <= RADIUS_KM).sum()), "T5_hits": int((final.T5_error_km <= RADIUS_KM).sum()),
        "methods": methods, "comparisons": comparisons,
        "frozen_sha256": {k: sha256(Path(bundle_assets[k])) for k in ("gate", "t4_confidence", "t5_confidence")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "routing"
        stage.mkdir()
        joblib.dump({"gate": gate, "confidence": calibrators}, stage / "regional_gate.joblib", compress=3)
        decisions.to_parquet(stage / "decisions.parquet", index=False)
        (stage / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    return report
