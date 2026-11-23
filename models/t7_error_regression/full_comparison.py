"""Compare the original and full-data T7 decisions on each archived evaluation photo."""
import numpy as np

from models.clip_regression.scripts.compare_t1 import _gps, _validate_ids
from models.t6_router.coverage_gate import ERRORS
from models.t6_router.evaluate import TEST_SPLITS
from models.t6_router.expanded_regions import FINAL
from models.t6_router.geographic_gate import POSITION, _errors
from models.t6_router.region_gate import paired_change
from models.t6_router.regional_closer import _choices, distance_metrics, paired_distances


COMMON = ("split", "block", *POSITION, *ERRORS)
EXPECTED = ("T4_expected_km", "T5_expected_km")
TRUTH = ("latitude", "longitude")


def _validate(frame, name):
    if not frame.columns.is_unique or not {"id", *COMMON, *EXPECTED, "T7_choose_T4"}.issubset(frame.columns):
        raise ValueError(f"{name}: missing or ambiguous decision columns")
    _validate_ids(frame, name)
    if frame.id.le(0).any() or frame.block.isna().any() or set(frame.split) != {FINAL, *TEST_SPLITS}:
        raise ValueError(f"{name}: expected real photo IDs, geographic blocks and all four evaluation cohorts")
    for expert, coords in (("T4", POSITION[:2]), ("T5", POSITION[2:])):
        _gps(frame, coords, expert)
    if any(col in frame for col in TRUTH):
        _gps(frame, TRUTH, name)
    _errors(frame)

    forecasts = frame.loc[:, list(EXPECTED)].to_numpy(dtype=float)
    picks = _choices(frame, frame.T7_choose_T4.to_numpy())
    if not np.isfinite(forecasts).all() or (forecasts < 0).any():
        raise ValueError(f"{name}: expected finite nonnegative error forecasts")
    if not np.array_equal(picks, forecasts[:, 0] < forecasts[:, 1]):
        raise ValueError(f"{name}: archived choice disagrees with smaller expected error or T5 tie rule")
    for suffix in ("lat_pred", "lon_pred", "error_km"):
        stored = f"T7_{suffix}"
        if stored in frame and not np.array_equal(frame[stored], np.where(picks, frame[f"T4_{suffix}"], frame[f"T5_{suffix}"])):
            raise ValueError(f"{name}: {stored} disagrees with selected expert")
    return frame.set_index("id").sort_index()


def _summarize(rows):
    new = rows.T7_choose_T4.to_numpy()
    old = rows.original_choose_T4.to_numpy()
    paired = paired_distances(rows, new, old)
    paired["unchanged"] = len(rows) - paired["improved"] - paired["worsened"]
    paired["hit25_changes"] = paired_change(rows, new, old)
    return {"n": len(rows), "original": distance_metrics(rows, old), "full": distance_metrics(rows, new),
            "paired": paired}


def compare_decisions(original, full):
    """Align archive-format decision frames by ID; report the three official tests and the separate reserve.

    Frozen expert positions and observed errors must agree exactly for each photo.
    Archived T7 tables store the two observed errors; actual GPS is also checked
    when both snapshots supply it. Saved kilometres are original error minus full error.
    """
    old, new = _validate(original, "original T7"), _validate(full, "full T7")
    if not old.index.equals(new.index):
        raise ValueError("Original and full T7 evaluation photo IDs differ")
    fields = [*COMMON, *(col for col in (*TRUTH, "T6_choose_T4") if col in old and col in new)]
    for field in fields:
        if not np.array_equal(old[field].to_numpy(), new[field].to_numpy()):
            raise ValueError(f"{field} differs for the same evaluation IDs")
    paired = new.assign(original_choose_T4=old.T7_choose_T4)
    official = paired.loc[paired.split.isin(TEST_SPLITS)]
    return {
        "new_final": _summarize(paired.loc[paired.split.eq(FINAL)]),
        "official": {"splits": {name: _summarize(official.loc[official.split.eq(name)]) for name in TEST_SPLITS},
                     "pooled": _summarize(official)},
    }
