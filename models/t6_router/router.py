"""Choose the coordinates of the more confident expert for the 25 km hit event."""

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
