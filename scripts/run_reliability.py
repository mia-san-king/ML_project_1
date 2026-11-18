#!/usr/bin/env python3
"""Train reliability model on val; evaluate on test splits (A4_tau5 + B2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.baseline_infer import load_feature_tables
from lib.geo import haversine_km
from lib.paths import FEATS, RESULTS
from lib.reliability import fit_reliability, report_classifier, risk_coverage

THRESHOLDS = (1, 5, 25)
TEST_SPLITS = ("test_in", "test_city", "test_region")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "report" / "figures"


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def spread_km(top_idx: np.ndarray, top_logit: np.ndarray, gallery: np.ndarray, pred_lat, pred_lon) -> np.ndarray:
    """Probability-weighted distance from the top-1 point to the other top-k candidates."""
    gps = gallery[top_idx]
    probs = softmax_rows(top_logit)
    lat = np.asarray(pred_lat, dtype=np.float64)[:, None]
    lon = np.asarray(pred_lon, dtype=np.float64)[:, None]
    dist = haversine_km(lat, lon, gps[:, :, 0], gps[:, :, 1])
    return (probs * dist).sum(axis=1)


def build_frame(run_name: str = "A4_tau5") -> pd.DataFrame:
    topk = np.load(RESULTS / "preds" / f"{run_name}_topk.npz", allow_pickle=True)
    gallery = np.load(FEATS / "gallery_grid1km_gps.npy")
    a4 = pd.read_parquet(RESULTS / "preds" / f"{run_name}.parquet")
    b2 = pd.read_parquet(RESULTS / "preds" / "B2.parquet")
    _, _, table = load_feature_tables()
    truth = table[["id", "latitude", "longitude", "view_type", "block"]]

    feat = pd.DataFrame(
        {
            "id": topk["ids"].astype(np.int64),
            "split": topk["split"].astype(str),
            "logsumexp": topk["logsumexp"].astype(np.float64),
        }
    )
    b2sub = b2[["id", "knn_sim", "lat_pred", "lon_pred"]].rename(
        columns={"lat_pred": "b2_lat", "lon_pred": "b2_lon"}
    )
    frame = feat.merge(a4, on=["id", "split"]).merge(b2sub, on="id").merge(truth, on="id")
    pos = pd.Series(np.arange(len(topk["ids"])), index=topk["ids"].astype(np.int64)).loc[frame.id.values].to_numpy()
    logits = topk["top_logit"][pos]
    lse = topk["logsumexp"][pos]
    frame["p1"] = np.exp(logits[:, 0] - lse)
    frame["margin"] = logits[:, 0] - logits[:, 1]
    frame["entropy_k"] = entropy(softmax_rows(logits))
    print(f"features n={len(frame)}", flush=True)
    frame["spread_k"] = spread_km(
        topk["top_idx"][pos], logits, gallery, frame.lat_pred.values, frame.lon_pred.values
    )
    frame["knn_agree"] = haversine_km(
        frame.lat_pred, frame.lon_pred, frame.b2_lat, frame.b2_lon
    )
    frame["err_km"] = haversine_km(frame.latitude, frame.longitude, frame.lat_pred, frame.lon_pred)

    vt = pd.get_dummies(frame["view_type"].fillna("unknown"), prefix="vt")
    frame = pd.concat([frame, vt], axis=1)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    base = ["p1", "margin", "entropy_k", "spread_k", "knn_sim", "knn_agree"]
    return base + [c for c in frame.columns if c.startswith("vt_")]


class _IsoP1:
    """Map scalar p1 to P(error <= T) with isotonic regression."""

    def fit(self, X, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(np.asarray(X, dtype=float).ravel(), np.asarray(y))
        return self

    def predict_proba(self, X):
        p = np.clip(self.iso.predict(np.asarray(X, dtype=float).ravel()), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def fit_p1_baseline(X_p1: np.ndarray, err: np.ndarray, T: float):
    y = (err <= T).astype(np.int32)
    return _IsoP1().fit(X_p1.reshape(-1, 1), y)


def plot_risk_coverage(conf: dict, err: np.ndarray, out_path: Path, T: float = 25):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for name, c in conf.items():
        cov, acc = risk_coverage(c, err, T)
        plt.plot(cov, acc * 100, label=name)
    plt.xlabel("coverage")
    plt.ylabel(f"accuracy @ {T:g} km (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_calibration(probs: np.ndarray, labels: np.ndarray, out_path: Path, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    centers, rates = [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if not mask.any():
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        rates.append(labels[mask].mean())
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.plot(centers, rates, "o-", label="model")
    plt.xlabel("predicted P(error <= T)")
    plt.ylabel("empirical")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    frame = build_frame("A4_tau5")
    cols = feature_columns(frame)
    val = frame[frame.split == "val"]
    X_val = val[cols].to_numpy(dtype=np.float32)
    groups = val["block"].values

    results = {"thresholds_km": list(THRESHOLDS), "features": cols, "test": {}}

    clf_by_T = {}
    p1_by_T = {}
    for T in THRESHOLDS:
        print(f"fit T={T} n_val={len(val)}", flush=True)
        clf_by_T[T] = fit_reliability(X_val, val.err_km.values, T, groups=groups)
        p1_by_T[T] = fit_p1_baseline(val.p1.values, val.err_km.values, T)
        print(f"fit T={T} done", flush=True)

    for split in TEST_SPLITS:
        sub = frame[frame.split == split]
        X = sub[cols].to_numpy(dtype=np.float32)
        err = sub.err_km.values
        results["test"][split] = {}
        for T in THRESHOLDS:
            results["test"][split][str(T)] = {
                "model": report_classifier(clf_by_T[T], X, err, T),
                "p1": report_classifier(p1_by_T[T], sub.p1.values.reshape(-1, 1), err, T),
            }

    T_main = 25.0
    test_in = frame[frame.split == "test_in"]
    conf = {
        "model": clf_by_T[T_main].predict_proba(test_in[cols].to_numpy(dtype=np.float32))[:, 1],
        "p1": p1_by_T[T_main].predict_proba(test_in.p1.values.reshape(-1, 1))[:, 1],
    }
    plot_risk_coverage(conf, test_in.err_km.values, FIG_DIR / "risk_coverage.png", T_main)

    for split, fname in [("test_in", "calibration_test_in.png"), ("test_region", "calibration_test_region.png")]:
        sub = frame[frame.split == split]
        p = clf_by_T[T_main].predict_proba(sub[cols].to_numpy(dtype=np.float32))[:, 1]
        y = (sub.err_km.values <= T_main).astype(float)
        plot_calibration(p, y, FIG_DIR / fname)

    out = RESULTS / "metrics" / "reliability.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    print(
        "test_in T=25 AUROC model",
        results["test"]["test_in"]["25"]["model"].get("auroc"),
        "p1",
        results["test"]["test_in"]["25"]["p1"].get("auroc"),
    )


if __name__ == "__main__":
    main()