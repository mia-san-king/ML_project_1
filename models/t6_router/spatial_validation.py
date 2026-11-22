"""Compare two distance signals with the existing 25 km router on validation blocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor, export_text
from threadpoolctl import threadpool_limits

from lib.confidence import GeoConfidence
from models.t6_router.router import choose_t4
from scripts.confidence.evaluate import FEATURES, RADIUS_KM, SEED


DISTANCES = ("expert_gap_km", "candidate_spread_km")


def _distances(values, frame, name):
    if isinstance(values, pd.Series) and not values.index.equals(frame.index):
        raise ValueError(f"{name} series index must match the frame")
    arr = np.asarray(values, dtype=float)
    if arr.shape != (len(frame),) or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError(f"Expected one finite, nonnegative {name} per validation photo")
    return arr


def _paired(candidate_hits, baseline_hits, blocks):
    made = candidate_hits & ~baseline_hits
    lost = baseline_hits & ~candidate_hits
    delta = made.astype(int) - lost.astype(int)
    groups, inverse = np.unique(blocks, return_inverse=True)
    changes = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(SEED).integers(len(groups), size=(3000, len(groups)))
    pp = 100 * changes[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "baseline_hits": int(baseline_hits.sum()), "candidate_hits": int(candidate_hits.sum()),
        "rescued": int(made.sum()), "lost": int(lost.sum()), "net": int(delta.sum()),
        "delta_pp": float(100 * delta.mean()),
        "block_ci95_pp": np.quantile(pp, [0.025, 0.975]).tolist(),
        "blocks_improved": int((changes > 0).sum()), "blocks_worse": int((changes < 0).sum()),
        "blocks_tied": int((changes == 0).sum()),
    }


def _refit_confidence(frame, train, hold, blocks):
    train_blocks = set(blocks[train])
    probabilities = {}
    for model in ("T4", "T5"):
        features = FEATURES[model]
        estimator = GeoConfidence(features, threshold_km=RADIUS_KM, random_state=SEED)
        fit = frame.iloc[train]
        estimator.fit(fit[features], fit[f"{model}_error_km"], groups=blocks[train], query_ids=fit.id)
        fitted, calibrated = set(estimator.fit_groups_), set(estimator.calibration_groups_)
        if fitted & calibrated or fitted | calibrated != train_blocks:
            raise RuntimeError(f"{model} confidence partition differs from the outer training fold")
        probabilities[model] = estimator.predict_proba(frame.iloc[hold][features])
    return probabilities


def evaluate(frame, gaps_km, spreads_km):
    """Return a JSON-ready report and ID-keyed out-of-fold predictions in input order.

    Each tree estimates the expected change in the 25 km hit indicator from
    switching to T4. Only a strictly positive predicted change selects T4.
    The two saved confidence probabilities are used solely for the archived
    T6 comparison. The same-fold T6 comparison refits its two calibrators.
    """
    if (not frame.columns.is_unique or frame[["id", "block"]].isna().any().any()
            or frame.id.duplicated().any()):
        raise ValueError("Validation requires unique photo IDs, unambiguous columns and known blocks")
    if "split" in frame and not frame.split.eq("val").all():
        raise ValueError("Only validation photos may enter this experiment")
    groups = frame.block.astype(str).to_numpy()
    if len(np.unique(groups)) < 5:
        raise ValueError("At least five geographic blocks are required")
    x = np.column_stack([_distances(values, frame, name) for name, values in
                         zip(DISTANCES, (gaps_km, spreads_km))])
    errors = frame[["T4_error_km", "T5_error_km"]].to_numpy(dtype=float)
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Each frozen expert needs a nonnegative finite error per photo")
    h4, h5 = (errors <= RADIUS_KM).T
    benefit = h4.astype(int) - h5.astype(int)
    archived = choose_t4(frame.T4_p25.to_numpy(), frame.T5_p25.to_numpy())

    fold_ids = np.full(len(frame), -1, dtype=int)
    predicted_gain = np.full(len(frame), np.nan)
    p4, p5 = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    folds = []
    with threadpool_limits(limits=4):
        for fold, (train, hold) in enumerate(GroupKFold(n_splits=5).split(x, groups=groups), start=1):
            held_blocks = sorted(set(groups[hold]))
            if set(groups[train]).intersection(held_blocks) or (fold_ids[hold] != -1).any():
                raise RuntimeError("Outer fold overlaps an evaluation block or repeats an evaluated photo")
            tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=250, random_state=SEED)
            tree.fit(x[train], benefit[train])
            fold_ids[hold] = fold
            predicted_gain[hold] = tree.predict(x[hold])
            prob = _refit_confidence(frame, train, hold, groups)
            p4[hold], p5[hold] = prob["T4"], prob["T5"]
            picks = predicted_gain[hold] > 0
            folds.append({
                "fold": fold, "n_train": len(train), "n_held": len(hold), "held_blocks": held_blocks,
                "choose_T4": int(picks.sum()), "all_held_choose_T5": bool(not picks.any()),
                "positive_training_leaves": int(np.sum(tree.tree_.value[:, 0, 0][tree.tree_.children_left == -1] > 0)),
                "rule": export_text(tree, feature_names=list(DISTANCES), decimals=5),
            })

    if (fold_ids < 1).any() or not np.isfinite(predicted_gain).all():
        raise RuntimeError("Some validation photos lack held-out predictions")
    candidate = predicted_gain > 0
    refit = choose_t4(p4, p5)
    results = pd.DataFrame({
        "id": frame.id.to_numpy(), "block": groups, "fold": fold_ids,
        DISTANCES[0]: x[:, 0], DISTANCES[1]: x[:, 1],
        "T4_error_km": errors[:, 0], "T5_error_km": errors[:, 1],
        "T4_archived_p25": frame.T4_p25.to_numpy(), "T5_archived_p25": frame.T5_p25.to_numpy(),
        "T4_refit_p25": p4, "T5_refit_p25": p5,
        "predicted_hit_gain": predicted_gain, "candidate_choose_T4": candidate,
        "archived_T6_choose_T4": archived, "block_refit_T6_choose_T4": refit,
    })
    policies = {"candidate": candidate, "archived_T6": archived, "block_refit_T6": refit,
                "always_T5": np.zeros(len(frame), dtype=bool), "always_T4": np.ones(len(frame), dtype=bool)}
    hits = {name: np.where(use4, h4, h5) for name, use4 in policies.items()}
    comparisons = {f"vs_{name}": _paired(hits["candidate"], hits[name], groups)
                   for name in ("archived_T6", "block_refit_T6", "always_T5")}
    for fold in folds:
        mask = fold_ids == fold["fold"]
        for name in ("archived_T6", "block_refit_T6", "always_T5"):
            baseline, new = hits[name][mask], hits["candidate"][mask]
            fold[f"vs_{name}"] = {"rescued": int((new & ~baseline).sum()), "lost": int((baseline & ~new).sum())}

    report = {
        "n": len(frame), "n_blocks": len(np.unique(groups)), "radius_km": RADIUS_KM,
        "design": {"outer_folds": 5, "splitter": "GroupKFold, no shuffle, spatial block",
                   "tree_max_depth": 2, "tree_min_samples_leaf": 250, "seed": SEED,
                   "target": "I(T4 error <= 25) - I(T5 error <= 25)", "choose_T4_when": "predicted gain > 0",
                   "bootstrap": "3,000 paired whole-block resamples of fixed out-of-fold predictions, photo-weighted"},
        "methods": {name: {"choose_T4": int(use4.sum()), "hits": int(hits[name].sum()),
                           "hit_pct": float(100 * hits[name].mean())} for name, use4 in policies.items()},
        "opportunities": {"only_T4_hits": int((h4 & ~h5).sum()), "only_T5_hits": int((h5 & ~h4).sum()),
                          "both_hit": int((h4 & h5).sum()), "neither_hits": int((~h4 & ~h5).sum()),
                          "oracle_hits": int((h4 | h5).sum())},
        "comparisons": comparisons, "folds": folds,
    }
    return report, results
