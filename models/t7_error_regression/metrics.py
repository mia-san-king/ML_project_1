"""Evaluate fixed T7/T6 choices and each expert's expected geographic error.

Closer-choice accuracy excludes exact expert-error ties; geographic errors and
the inclusive 25 km hit rate retain every photo. Calibration bins condition on
predicted expected km, are left-closed/right-open, and include 100 in 100+.
"""
import numpy as np

from models.t6_router.geographic_gate import _errors
from models.t6_router.region_gate import paired_change
from models.t6_router.regional_closer import _choices, distance_metrics, paired_distances


CALIBRATION_BINS = (
    (0., 10., "0-10"), (10., 25., "10-25"), (25., 50., "25-50"),
    (50., 100., "50-100"), (100., None, "100+"),
)
REQUIRED = (
    "T4_error_km", "T5_error_km", "T4_expected_km", "T5_expected_km",
    "T7_choose_T4", "T6_choose_T4",
)


def _expected_error(predicted, observed):
    bins = []
    for lower, upper, name in CALIBRATION_BINS:
        keep = predicted >= lower
        if upper is not None:
            keep &= predicted < upper
        count = int(keep.sum())
        bins.append({
            "bin": name, "lower_km": lower, "upper_km": upper, "count": count,
            "mean_predicted_km": float(predicted[keep].mean()) if count else None,
            "mean_observed_km": float(observed[keep].mean()) if count else None,
        })
    return {
        "mae_km": float(np.abs(predicted - observed).mean()),
        "mean_predicted_km": float(predicted.mean()),
        "mean_observed_km": float(observed.mean()),
        "calibration_bins": bins,
    }


def evaluate(frame):
    """Return JSON-ready diagnostics for one nonempty, already aligned cohort.

    The six REQUIRED columns contain finite nonnegative errors/expectations and
    boolean choices (True chooses T4). No fitting or routing decisions occur.
    Empty calibration bins and undefined all-tie closer accuracy use None.
    Positive paired saved km and net hits mean improvement over T6.
    """
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Evaluation needs nonempty rows and unique columns")
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
    observed = _errors(frame)
    predicted = frame[["T4_expected_km", "T5_expected_km"]].to_numpy(dtype=float)
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Expected errors must be finite and nonnegative km")
    t7 = _choices(frame, frame.T7_choose_T4.to_numpy())
    t6 = _choices(frame, frame.T6_choose_T4.to_numpy())
    policies = {
        "T7": t7, "T6": t6,
        "always_T4": np.ones(len(frame), dtype=bool),
        "always_T5": np.zeros(len(frame), dtype=bool),
    }
    methods = {name: distance_metrics(frame, picks) for name, picks in policies.items()}
    paired = paired_distances(frame, t7, t6)
    paired["hit25_changes"] = paired_change(frame, t7, t6)
    return {
        "n": len(frame),
        "methods": methods,
        "expected_km": {expert: _expected_error(predicted[:, index], observed[:, index])
                        for index, expert in enumerate(("T4", "T5"))},
        "paired_vs_T6": paired,
    }
