"""Generate out-of-block expert confidences for training the joint T4/T5 selector."""
import numpy as np
from sklearn.model_selection import GroupKFold

from models.t6_router.spatial_validation import _refit_confidence


def crossfit_confidence(frame, *, n_splits=4):
    """For each row return T4,T5 probabilities from calibrators fitted on other blocks."""
    if frame.block.isna().any() or frame.id.duplicated().any():
        raise ValueError("Confidence cross-fitting needs unique photos with known blocks")
    blocks = frame.block.astype(str).to_numpy()
    scores = np.full((len(frame), 2), np.nan)
    for train, held in GroupKFold(n_splits=n_splits).split(frame, groups=blocks):
        probs = _refit_confidence(frame, train, held, blocks)
        scores[held] = np.column_stack([probs["T4"], probs["T5"]])
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RuntimeError("Missing or invalid held-out confidence scores")
    return scores
