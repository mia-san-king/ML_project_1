"""Route expert predictions with the joint T6 gate or the archived comparison rule."""

import joblib
import numpy as np


def choose_t4(p4, p5):
    """Return a boolean mask in query order. Exact ties keep T5."""
    p4 = np.asarray(p4, dtype=float)
    p5 = np.asarray(p5, dtype=float)
    if p4.ndim != 1 or p4.shape != p5.shape:
        raise ValueError("Expected aligned one-dimensional T4 and T5 probability arrays")
    for values in (p4, p5):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Each confidence must be a finite probability in [0, 1]")
    return p4 > p5


def select_expert(frame, gate_path=None, *, legacy_confidence=False):
    """Return the T4 mask and, for the learned gate, the conditional T4 probabilities."""
    if legacy_confidence:
        if gate_path is not None:
            raise ValueError("Choose either a learned gate or --legacy-confidence")
        return choose_t4(frame.T4_p25, frame.T5_p25), None
    if gate_path is None:
        raise ValueError("Expected a v2 T6 bundle with gate.joblib; upgrade the v1 bundle or specify --legacy-confidence")
    from models.t6_router.geographic_gate import NONPOSITION, POSITION

    features = frame.loc[:, list(POSITION + NONPOSITION)]
    probability = joblib.load(gate_path).predict_t4_probability(features)
    return probability > 0.5, probability
