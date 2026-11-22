"""Learn to choose a frozen location expert from its coordinates and confidence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from models.t6_router.gate_confidence import crossfit_confidence
from models.t6_router.router import choose_t4
from models.t6_router.spatial_validation import _paired
from scripts.confidence.evaluate import RADIUS_KM, SEED


POSITION = ("T4_lat_pred", "T4_lon_pred", "T5_lat_pred", "T5_lon_pred")
NONPOSITION = ("T4_p25", "T5_p25")
PARAMETERS = dict(max_iter=80, max_leaf_nodes=7, min_samples_leaf=100,
                  learning_rate=0.08, l2_regularization=1.0, early_stopping=False,
                  random_state=SEED)


def _features(frame, columns):
    if not frame.columns.is_unique:
        raise ValueError("Ambiguous feature columns")
    x = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("The gate requires finite expert coordinates and probabilities")
    for name, values in zip(columns, x.T):
        if name in NONPOSITION:
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"{name} must be a probability in [0, 1]")
        elif (np.abs(values) > (90 if name.endswith("_lat_pred") else 180)).any():
            raise ValueError(f"{name} must contain valid WGS84 coordinates")
    return x


def _errors(frame):
    values = frame[["T4_error_km", "T5_error_km"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expert errors must be finite and nonnegative")
    return values


@dataclass
class GeographicGate:
    """Probability that T4 wins, conditional on exactly one expert hitting."""

    feature_names: tuple[str, ...]
    estimator: HistGradientBoostingClassifier | None
    fallback: float

    def predict_t4_probability(self, frame):
        x = _features(frame, self.feature_names)
        if self.estimator is None or not len(x):
            return np.full(len(x), self.fallback)
        return self.estimator.predict_proba(x)[:, 1]


def fit_gate(frame, use_position=True):
    """Fit with naturally occurring exclusive hits; ties contribute zero to the hit-rate difference.

    Training probabilities must have been generated out of fold.
    """
    columns = POSITION + NONPOSITION if use_position else NONPOSITION
    x = _features(frame, columns)
    h4, h5 = (_errors(frame) <= RADIUS_KM).T
    decisive = h4 != h5
    labels = h4[decisive].astype(int)
    if len(np.unique(labels)) < 2:
        fallback = float(labels[0]) if len(labels) else 0.5
        return GeographicGate(columns, None, fallback)
    with threadpool_limits(limits=4):
        model = HistGradientBoostingClassifier(**PARAMETERS).fit(x[decisive], labels)
    return GeographicGate(columns, model, 0.5)


def _aligned_decisions(frame, old):
    if (not frame.columns.is_unique or not old.columns.is_unique or
            frame[["id", "block"]].isna().any().any() or old[["id", "block", "fold"]].isna().any().any()):
        raise ValueError("Need distinct columns and complete photo IDs, blocks and fold assignments")
    if frame.id.duplicated().any() or old.id.duplicated().any() or set(old.id) != set(frame.id):
        raise ValueError("Saved decisions must contain one row for each validation ID")
    if "split" in frame and not frame.split.eq("val").all():
        raise ValueError("Only validation rows are allowed")
    old = old.set_index("id").loc[frame.id].reset_index()
    if not np.array_equal(old.block.astype(str), frame.block.astype(str)):
        raise ValueError("Archived and current geographic blocks do not match")
    if (set(old.fold) != {1, 2, 3, 4, 5}
            or not old.groupby(old.block.astype(str)).fold.nunique().eq(1).all()):
        raise ValueError("Expected five saved folds assigning each block to exactly one fold")
    for column in ("T4_error_km", "T5_error_km"):
        if column in old:
            np.testing.assert_allclose(old[column], frame[column], atol=1e-5, rtol=0)
    for column in ("archived_T6_choose_T4", "block_refit_T6_choose_T4"):
        if not old[column].isin([True, False]).all():
            raise ValueError(f"{column} must contain boolean decisions")
    np.testing.assert_array_equal(choose_t4(old.T4_refit_p25, old.T5_refit_p25), old.block_refit_T6_choose_T4)
    return old


def evaluate_gate(frame, old_decisions):
    """Cross-fit the two gates using the archived outer folds and honest confidence inputs."""
    old = _aligned_decisions(frame, old_decisions)
    e4, e5 = _errors(frame).T
    h4, h5 = e4 <= RADIUS_KM, e5 <= RADIUS_KM
    groups = frame.block.astype(str).to_numpy()
    fold_id = old.fold.to_numpy()
    held_p = old[["T4_refit_p25", "T5_refit_p25"]].to_numpy(dtype=float)
    use4 = {"archived_T6": old.archived_T6_choose_T4.to_numpy(dtype=bool),
            "block_refit_T6": old.block_refit_T6_choose_T4.to_numpy(dtype=bool),
            "always_T5": np.zeros(len(frame), dtype=bool), "always_T4": np.ones(len(frame), dtype=bool)}
    scores = {name: np.full(len(frame), np.nan) for name in ("geographic_gate", "without_position")}
    folds = []
    with threadpool_limits(limits=4):
        for fold in sorted(set(fold_id)):
            hold = fold_id == fold
            train = ~hold
            development = frame.loc[train].copy()
            held = frame.loc[hold].copy()
            development[list(NONPOSITION)] = crossfit_confidence(development, n_splits=4)
            held[list(NONPOSITION)] = held_p[hold]
            for name in scores:
                model = fit_gate(development, use_position=name == "geographic_gate")
                scores[name][hold] = model.predict_t4_probability(held)
            folds.append({"fold": int(fold), "n_train": int(train.sum()), "n_held": int(hold.sum()),
                          "n_train_discordant": int((h4[train] != h5[train]).sum()),
                          "held_blocks": sorted(set(groups[hold]))})

    decisions = frame[["id", "block", *POSITION, "T4_error_km", "T5_error_km"]].copy()
    decisions["fold"] = fold_id
    decisions[["T4_refit_p25", "T5_refit_p25"]] = held_p
    for name, score in scores.items():
        if not np.isfinite(score).all():
            raise RuntimeError(f"Missing held-out probability for {name}")
        decisions[f"{name}_p4"] = score
        use4[name] = score > 0.5
    metrics, hits = {}, {}
    for name, select in use4.items():
        distances = np.where(select, e4, e5)
        hits[name] = distances <= RADIUS_KM
        metrics[name] = {"choose_T4": int(select.sum()), "hits": int(hits[name].sum()),
                         "hit_pct": float(100 * hits[name].mean()), "median_km": float(np.median(distances))}
        decisions[f"{name}_choose_T4"] = select
    best = hits["geographic_gate"]
    comparisons = {f"vs_{name}": _paired(best, hits[name], groups) for name in
                   ("archived_T6", "block_refit_T6", "always_T5", "without_position")}
    for fold in folds:
        mask = fold_id == fold["fold"]
        fold["choose_T4"] = int(use4["geographic_gate"][mask].sum())
        fold["hits"] = int(best[mask].sum())
        fold["comparisons"] = {}
        for key in comparisons:
            baseline = hits[key.removeprefix("vs_")][mask]
            saved = int((best[mask] & ~baseline).sum())
            lost = int((~best[mask] & baseline).sum())
            fold["comparisons"][key] = {"rescued": saved, "lost": lost, "net": saved - lost}
    summary = {
        "n": len(frame), "n_blocks": len(set(groups)), "radius_km": RADIUS_KM,
        "design": {"features": list(POSITION + NONPOSITION), "ablation_features": list(NONPOSITION),
                   "outer_folds": 5, "inner_confidence_folds": 4, "estimator": "HistGradientBoostingClassifier",
                   "parameters": PARAMETERS.copy(), "target": "T4 hit conditional on exactly one expert hitting",
                   "select_T4_when": "conditional T4 probability > 0.5",
                   "bootstrap": "3,000 paired spatial block resamples of fixed outer-fold predictions"},
        "opportunities": {"only_T4_hits": int((h4 & ~h5).sum()), "only_T5_hits": int((~h4 & h5).sum()),
                          "both_hit": int((h4 & h5).sum()), "neither_hit": int((~h4 & ~h5).sum()),
                          "oracle_hits": int((h4 | h5).sum())},
        "methods": metrics, "comparisons": comparisons, "folds": folds,
    }
    return summary, decisions
