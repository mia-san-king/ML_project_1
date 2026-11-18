import numpy as np
import pandas as pd

from lib.geo import haversine_km

THRESHOLDS_KM = (1, 5, 10, 25, 50, 200, 750, 2500)


def summarize(err_km):
    errors = np.asarray(err_km, dtype=float)
    out = {
        "n": int(len(errors)),
        "median_km": float(np.median(errors)),
        "mean_km": float(errors.mean()),
        "p90_km": float(np.percentile(errors, 90)),
    }
    out.update({f"acc@{threshold}km": float((errors <= threshold).mean() * 100) for threshold in THRESHOLDS_KM})
    return out


def bootstrap_ci(err_km, blocks, stat=np.median, n=1000, seed=0):
    """Resample spatial blocks (not individual images)."""
    rng = np.random.default_rng(seed)
    errors = np.asarray(err_km, dtype=float)
    block_ids = np.asarray(blocks)
    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(len(unique_blocks))]
    values = []
    for _ in range(n):
        picked = rng.integers(0, len(unique_blocks), len(unique_blocks))
        sample_idx = np.concatenate([groups[k] for k in picked])
        values.append(stat(errors[sample_idx]))
    return np.percentile(values, [2.5, 97.5])


def evaluate(pred: pd.DataFrame, truth: pd.DataFrame):
    merged = pred.merge(
        truth[["id", "latitude", "longitude", "view_type", "block"]],
        on="id",
        how="inner",
    )
    merged["err_km"] = haversine_km(
        merged.latitude.values,
        merged.longitude.values,
        merged.lat_pred.values,
        merged.lon_pred.values,
    )
    rows = []
    for split_name, group in merged.groupby("split"):
        rows.append(
            {
                "split": split_name,
                "view_type": "all",
                **summarize(group.err_km.values),
                "median_ci": bootstrap_ci(group.err_km.values, group.block.values).tolist(),
            }
        )
        for view_type, subset in group.groupby(group.view_type.fillna("oblique/none")):
            rows.append(
                {
                    "split": split_name,
                    "view_type": view_type,
                    **summarize(subset.err_km.values),
                }
            )
    return pd.DataFrame(rows), merged