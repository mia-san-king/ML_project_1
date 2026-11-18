"""Reliability / calibration helpers for geo predictions."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        total += abs(probs[mask].mean() - labels[mask].mean()) * mask.mean()
    return float(total)


def risk_coverage(conf: np.ndarray, err_km: np.ndarray, threshold_km: float = 25.0):
    conf = np.asarray(conf, dtype=float)
    err = np.asarray(err_km, dtype=float)
    order = np.argsort(-conf)
    e = err[order]
    n = len(e)
    cov = np.arange(1, n + 1, dtype=float) / n
    hits = np.cumsum(e <= threshold_km) / np.arange(1, n + 1, dtype=float)
    return cov, hits


def fit_reliability(X: np.ndarray, err_km: np.ndarray, T: float, groups: np.ndarray | None = None):
    y = (np.asarray(err_km) <= T).astype(np.int32)
    base = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=0)
    if groups is not None:
        gkf = GroupKFold(n_splits=5)
        best_depth, best_auroc = 4, -1.0
        for depth in (2, 4, 6):
            scores = []
            for tr, te in gkf.split(X, y, groups):
                clf = HistGradientBoostingClassifier(
                    max_depth=depth, learning_rate=0.05, max_iter=200, random_state=0
                )
                clf.fit(X[tr], y[tr])
                p = clf.predict_proba(X[te])[:, 1]
                if len(np.unique(y[te])) < 2:
                    continue
                scores.append(roc_auc_score(y[te], p))
            if scores and np.mean(scores) > best_auroc:
                best_auroc, best_depth = float(np.mean(scores)), depth
        base = HistGradientBoostingClassifier(
            max_depth=best_depth, learning_rate=0.05, max_iter=200, random_state=0
        )

    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(X, y)
    return clf


def report_classifier(clf, X: np.ndarray, err_km: np.ndarray, T: float) -> dict:
    y = (np.asarray(err_km) <= T).astype(np.int32)
    p = clf.predict_proba(X)[:, 1]
    out = {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(p, y),
    }
    if len(np.unique(y)) >= 2:
        out["auroc"] = float(roc_auc_score(y, p))
    else:
        out["auroc"] = None
    return out