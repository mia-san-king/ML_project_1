"""Compare geographic coverage of closer-expert training at matched photo budgets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from models.t6_router.geographic_gate import NONPOSITION, PARAMETERS, POSITION, _errors
from models.t6_router.region_gate import _validate_partition, confidence_scores, fit_confidence, paired_change
from models.t6_router.regional_closer import distance_metrics, fit_closer, paired_distances
from scripts.confidence.evaluate import FEATURES, SEED, sha256


THREE, SEVEN = "three_regions", "seven_regions"
RAW = FEATURES["T4"] + FEATURES["T5"]
ERRORS = ["T4_error_km", "T5_error_km"]
USED = ["id", "region", "block", *POSITION, *RAW, *ERRORS]


def _signature(frame, columns):
    ordered = frame.sort_values("id").loc[:, columns]
    values = pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype="<u8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _counts(rows):
    return {str(key): int(n) for key, n in rows.region.value_counts().sort_index().items()}


def _balanced(rows, budget):
    """Draw from available regions round-robin, with deterministic within-region order."""
    ordered = rows.sort_values("id")
    if budget == len(ordered):
        return ordered
    permuted = ordered.sample(frac=1, random_state=SEED)
    within = permuted.groupby("region", sort=False).cumcount()
    return permuted.iloc[np.argsort(within.to_numpy(), kind="stable")[:budget]].sort_values("id")


def _validate(three, seven, final):
    for expected, frame in ((3, three), (7, seven)):
        _validate_partition(frame, final)
        if frame.region.nunique() != expected:
            raise ValueError(f"Expected exactly {expected} distinct development regions")
        if frame.groupby("block").region.nunique().gt(1).any():
            raise ValueError("A development block belongs to multiple held-out regions")
        if set(frame.block.astype(str)) & set(final.block.astype(str)):
            raise ValueError("Development and final geographic blocks overlap")
    if len(three) != len(seven):
        raise ValueError("The two training samples must have the same number of photos")
    if not set(three.region).issubset(seven.region):
        raise ValueError("The broader sample must cover the original three regions")
    left, right = (data[USED].set_index("id", verify_integrity=True) for data in (three, seven))
    shared = left.index.intersection(right.index)
    try:
        pd.testing.assert_frame_equal(left.loc[shared], right.loc[shared], check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError("An overlapping photo has different expert data across the two samples") from exc


def _fit_arm(frame, confidence_budget, gate_budget):
    rows = frame.sort_values("id").reset_index(drop=True).copy()
    probabilities = np.full((len(rows), 2), np.nan)
    folds = {}
    for region in sorted(rows.region.unique()):
        held = rows.region.eq(region).to_numpy()
        source = _balanced(rows.loc[~held], confidence_budget)
        probabilities[held] = confidence_scores(rows.loc[held], fit_confidence(source))
        folds[str(region)] = {"held_n": int(held.sum()), "confidence_fit_n": len(source),
                              "confidence_source_regions": _counts(source), "source_ids_sha256": _signature(source, ["id"])}
    rows[list(NONPOSITION)] = probabilities
    e4, e5 = _errors(rows).T
    labeled = _balanced(rows.loc[e4 != e5], gate_budget)
    model = {"gate": fit_closer(labeled), "confidence": fit_confidence(rows)}
    info = {"photos_by_region": _counts(rows), "oof": folds, "source_ties": int((e4 == e5).sum()),
            "gate_train_by_region": _counts(labeled), "gate_train_n": len(labeled),
            "gate_train_ids_sha256": _signature(labeled, ["id"]), "final_confidence_train_n": len(rows),
            "sample_sha256": _signature(rows, USED)}
    return model, info


def _block_results(decisions, methods, changes):
    e3, e7 = decisions[f"{THREE}_error_km"], decisions[f"{SEVEN}_error_km"]
    unequal = decisions.T4_error_km.ne(decisions.T5_error_km)
    oracle = decisions.oracle_error_km
    observations = pd.DataFrame({
        "block": decisions.block.astype(str), "n": 1, "n_non_ties": unequal,
        "three_correct": unequal & e3.eq(oracle), "seven_correct": unequal & e7.eq(oracle),
        "improved": e7.lt(e3), "worsened": e7.gt(e3),
    })
    by_block = observations.groupby("block").sum().astype(int)
    by_block["net"] = by_block.improved - by_block.worsened
    if not by_block.net.eq(by_block.seven_correct - by_block.three_correct).all():
        raise RuntimeError("Geographic block transitions disagree with per-arm correct counts")
    expected = {"n": len(decisions), "n_non_ties": methods[SEVEN]["n_non_ties"],
                "three_correct": methods[THREE]["correct_non_ties"], "seven_correct": methods[SEVEN]["correct_non_ties"],
                "improved": changes["improved"], "worsened": changes["worsened"], "net": changes["net_correct_non_ties"]}
    if by_block.sum().to_dict() != expected:
        raise RuntimeError("Geographic block totals disagree with whole-region counts")
    return by_block.to_dict(orient="index")


def run_coverage_gate(three, seven, final, *, output_dir):
    """Train both arms on developer-supplied matched samples; score the final labels once."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Choose a new coverage output directory: {output}")
    _validate(three, seven, final)
    samples = {THREE: three, SEVEN: seven}
    # Every out-of-region confidence estimate sees the same number of labelled source photos.
    confidence_budget = min(len(f) - int(f.region.value_counts().max()) for f in samples.values())
    errors = [_errors(f) for f in samples.values()]
    gate_budget = min(int(np.count_nonzero(e[:, 0] != e[:, 1])) for e in errors)
    if gate_budget == 0:
        raise ValueError("Both samples need non-tied development labels to compare closer gates")
    final = final.reset_index(drop=True)
    inference = final[[*POSITION, *RAW]].copy()
    decisions = final[["id", "region", "block", *POSITION]].copy()
    models, training = {}, {}
    with threadpool_limits(limits=4):
        for arm, sample in samples.items():
            models[arm], training[arm] = _fit_arm(sample, confidence_budget, gate_budget)
            probabilities = confidence_scores(inference, models[arm]["confidence"])
            inputs = inference.loc[:, list(POSITION)].copy()
            inputs[list(NONPOSITION)] = probabilities
            p = models[arm]["gate"].predict_t4_probability(inputs)
            decisions[[f"{arm}_{name}" for name in NONPOSITION]] = probabilities
            decisions[f"{arm}_p_T4"] = p
            decisions[f"{arm}_choose_T4"] = p > .5

    # The final photo errors first enter after both arms have finished choosing.
    e4, e5 = _errors(final).T
    decisions[ERRORS] = final[ERRORS]
    decisions["oracle_error_km"] = np.minimum(e4, e5)
    methods = {}
    for arm in samples:
        picks = decisions[f"{arm}_choose_T4"].to_numpy()
        decisions[f"{arm}_error_km"] = np.where(picks, e4, e5)
        methods[arm] = distance_metrics(decisions, picks)
    methods["always_T4"] = distance_metrics(decisions, np.ones(len(final), dtype=bool))
    methods["always_T5"] = distance_metrics(decisions, np.zeros(len(final), dtype=bool))
    new, old = (decisions[f"{arm}_choose_T4"].to_numpy() for arm in (SEVEN, THREE))
    changes = paired_distances(decisions, new, old)
    changes["hit25_changes"] = paired_change(decisions, new, old)
    if (changes["net_correct_non_ties"] != methods[SEVEN]["correct_non_ties"] - methods[THREE]["correct_non_ties"]
            or changes["hit25_changes"]["net_hits_25km"] != methods[SEVEN]["hits_25km"] - methods[THREE]["hits_25km"]):
        raise RuntimeError("Paired transitions disagree with independent policy totals")
    report = {
        "n": len(final), "final_region": str(final.region.iloc[0]), "final_n_blocks": int(final.block.nunique()),
        "development_n_per_arm": len(three), "gate_train_n_per_arm": gate_budget,
        "oof_confidence_train_n_per_fold": confidence_budget, "training": training,
        "final_input_sha256": _signature(final, USED), "T4_closer": int((e4 < e5).sum()),
        "T5_closer": int((e5 < e4).sum()), "exact_ties": int((e4 == e5).sum()),
        "oracle_mean_km": float(decisions.oracle_error_km.mean()),
        "features": list(POSITION + NONPOSITION), "gate_parameters": PARAMETERS.copy(),
        "protocol": "Leave each source region out for its training confidences. Match the per-fold source "
                    "photo count across arms by deterministic region-balanced sampling. Refit both calibrators on "
                    "each arm's entire matched sample. Train the same unweighted closer gate; exclude exact ties "
                    "from its training and accuracy. Select T4 for score > 0.5.",
        "methods": methods, "seven_vs_three": changes, "final_by_block": _block_results(decisions, methods, changes),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "coverage"
        stage.mkdir()
        for arm, model in models.items():
            joblib.dump(model, stage / f"{arm}.joblib", compress=3)
        decisions.to_parquet(stage / "decisions.parquet", index=False)
        report["outputs_sha256"] = {file.name: sha256(file) for file in sorted(stage.iterdir())}
        (stage / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    return report
