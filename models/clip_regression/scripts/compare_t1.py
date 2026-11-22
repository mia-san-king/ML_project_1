"""Strict paired evaluation of T4 regression and the original T1 grid output.

Example (from any working directory)::

    python compare_t1.py --t4 t4.parquet --t1 original_t1.parquet --output comparison.json

Truth comes from T4's latitude/longitude, or --data-root/meta/smapshot.parquet.
Providing --data-root also checks embedded truth against that metadata. No model
or training data is loaded, and predictions are never silently intersected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


EVAL_SPLITS = ("val", "test_in", "test_city", "test_region")
ERROR_ATOL_KM = 1e-5
TIE_ATOL_KM = 1e-9


def _require_columns(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _validate_ids(frame, name):
    _require_columns(frame, ["id"], name)
    ids = frame["id"]
    if (frame.empty or ids.isna().any() or pd.api.types.is_bool_dtype(ids.dtype)
            or not pd.api.types.is_integer_dtype(ids.dtype)
            or ids.min() < np.iinfo(np.int64).min or ids.max() > np.iinfo(np.int64).max):
        raise ValueError(f"{name} requires nonempty, non-null integer IDs within int64 range")
    if ids.duplicated().any():
        raise ValueError(f"{name} contains duplicate IDs: {ids[ids.duplicated()].head().tolist()}")


def _gps(frame, columns, name):
    _require_columns(frame, columns, name)
    coordinates = frame[list(columns)].to_numpy(dtype=np.float64)
    if (not np.isfinite(coordinates).all()
            or np.any(np.abs(coordinates[:, 0]) > 90)
            or np.any(np.abs(coordinates[:, 1]) > 180)):
        raise ValueError(f"{name} coordinates must be finite WGS84 latitude/longitude")
    return coordinates


def _errors(truth, prediction):
    # Same Haversine definition/radius as lib.geo, without its data-dir side effects.
    lat1, lon1 = np.deg2rad(truth).T
    lat2, lon2 = np.deg2rad(prediction).T
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _metrics(errors):
    return {"median_km": float(np.median(errors)), "mean_km": float(errors.mean()),
            "acc@25km": float(100 * np.mean(errors <= 25)),
            "hits@25km": int(np.count_nonzero(errors <= 25))}


def compare_predictions(t4, t1, *, truth=None):
    """Return per-split metrics after strict ID/split and numeric validation.

    ``truth`` may contain extra metadata IDs; both prediction ID sets must match
    exactly. Saved errors are checked against float64 distances, never trusted.
    """
    for name, frame in (("T4", t4), ("T1", t1)):
        _validate_ids(frame, name)
        _require_columns(frame, ["split", "lat_pred", "lon_pred"], name)
        if frame["split"].isna().any() or not frame["split"].isin(EVAL_SPLITS).all():
            raise ValueError(f"{name} split must be one of {EVAL_SPLITS}")
        _gps(frame, ("lat_pred", "lon_pred"), name)
    _require_columns(t4, ["error_km"], "T4")
    if "method" in t1.columns or "neighbor_id" in t1.columns:
        raise ValueError("Use the original T1 grid predictions, not a retrieval/reranking artifact")

    ids4, ids1 = pd.Index(t4["id"]), pd.Index(t1["id"])
    missing, extra = ids4.difference(ids1), ids1.difference(ids4)
    if len(missing) or len(extra):
        raise ValueError("ID sets do not match: "
                         f"T4={len(t4)}, T1={len(t1)}; missing from T1={len(missing)} "
                         f"(examples {missing[:5].tolist()}), extra in T1={len(extra)} "
                         f"(examples {extra[:5].tolist()}). No intersection comparison performed.")
    t4 = t4.set_index("id", drop=False).sort_index()
    t1 = t1.set_index("id", drop=False).loc[t4.index]
    mismatch = t4["split"].to_numpy() != t1["split"].to_numpy()
    if np.any(mismatch):
        raise ValueError(f"split mismatch for {int(mismatch.sum())} IDs: {t4.index[mismatch][:5].tolist()}")

    truth_columns = ("latitude", "longitude")
    embedded = all(column in t4 for column in truth_columns)
    if truth is not None:
        _validate_ids(truth, "metadata truth")
        _require_columns(truth, truth_columns, "metadata truth")
        truth = truth.set_index("id")
        absent = t4.index.difference(truth.index)
        if len(absent):
            raise ValueError(f"metadata truth is missing {len(absent)} prediction IDs")
        gps_truth = _gps(truth.loc[t4.index], truth_columns, "metadata truth")
        truth_source = "metadata"
    elif embedded:
        gps_truth = _gps(t4, truth_columns, "T4 truth")
        truth_source = "T4 latitude/longitude (not independently checked against metadata)"
    else:
        raise ValueError("T4 needs latitude/longitude truth, or supply --data-root with meta/smapshot.parquet")
    for name, frame in (("T4", t4), ("T1", t1)):
        if any(column in frame for column in truth_columns):
            values = _gps(frame, truth_columns, f"{name} truth")
            if not np.allclose(values, gps_truth, atol=1e-9, rtol=0):
                raise ValueError(f"{name} embedded truth conflicts with comparison truth")

    errors = {}
    for name, frame in (("T4", t4), ("T1", t1)):
        errors[name] = _errors(gps_truth, _gps(frame, ("lat_pred", "lon_pred"), name))
        if "error_km" in frame:
            stored = frame["error_km"].to_numpy(dtype=np.float64)
            if (not np.isfinite(stored).all() or np.any(stored < 0)
                    or not np.allclose(stored, errors[name], atol=ERROR_ATOL_KM, rtol=0)):
                raise ValueError(f"{name} error_km differs from recomputed distances (atol={ERROR_ATOL_KM} km)")

    by_split = {}
    for split in EVAL_SPLITS:
        mask = t4["split"].to_numpy() == split
        if not np.any(mask):
            continue
        e4, e1 = errors["T4"][mask], errors["T1"][mask]
        m4, m1 = _metrics(e4), _metrics(e1)
        h4, h1 = e4 <= 25, e1 <= 25
        delta = e4 - e1
        by_split[split] = {
            "n": int(mask.sum()), "t4": m4, "t1": m1,
            "t4_minus_t1": {"median_km": m4["median_km"] - m1["median_km"],
                             "mean_km": m4["mean_km"] - m1["mean_km"],
                             "acc@25km_percentage_points": m4["acc@25km"] - m1["acc@25km"],
                             "hits@25km": m4["hits@25km"] - m1["hits@25km"]},
            "paired": {"t4_closer": int(np.count_nonzero(delta < -TIE_ATOL_KM)),
                       "t4_farther": int(np.count_nonzero(delta > TIE_ATOL_KM)),
                       "equal_error": int(np.count_nonzero(np.abs(delta) <= TIE_ATOL_KM)),
                       "t4_gained_hit25": int(np.count_nonzero(h4 & ~h1)),
                       "t4_lost_hit25": int(np.count_nonzero(~h4 & h1)),
                       "both_hit25": int(np.count_nonzero(h4 & h1)),
                       "both_missed25": int(np.count_nonzero(~h4 & ~h1))},
        }
    return {"comparison": "T4 regression versus original T1 grid",
            "alignment": {"mode": "strict identical ID sets and per-ID splits", "n": len(t4)},
            "truth_source": truth_source,
            "definitions": {"earth_radius_km": 6371.0088, "accuracy_unit": "percent",
                            "hit_threshold": "error_km <= 25", "delta_direction": "T4 minus T1",
                            "saved_error_atol_km": ERROR_ATOL_KM, "paired_tie_atol_km": TIE_ATOL_KM},
            "by_split": by_split}


def _provenance(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t4", required=True, type=Path, help="T4 predictions parquet, including error_km")
    parser.add_argument("--t1", required=True, type=Path, help="Original T1 grid predictions parquet")
    parser.add_argument("--output", required=True, type=Path, help="Comparison JSON to write")
    parser.add_argument("--data-root", type=Path, help="Optional data root containing meta/smapshot.parquet")
    args = parser.parse_args(argv)
    paths = {"t4": args.t4.expanduser().resolve(), "t1": args.t1.expanduser().resolve()}
    if args.data_root:
        paths["truth"] = (args.data_root.expanduser() / "meta/smapshot.parquet").resolve()
    output = args.output.expanduser().resolve()
    if output in paths.values():
        parser.error("--output must not overwrite an input file")
    try:
        truth = pd.read_parquet(paths["truth"], columns=["id", "latitude", "longitude"]) if "truth" in paths else None
        report = compare_predictions(pd.read_parquet(paths["t4"]), pd.read_parquet(paths["t1"]), truth=truth)
        report["inputs"] = {name: _provenance(path) for name, path in paths.items()}
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)
    for split, row in report["by_split"].items():
        print(f"{split}: n={row['n']}; T4/T1 median={row['t4']['median_km']:.3f}/{row['t1']['median_km']:.3f} km; "
              f"acc@25km={row['t4']['acc@25km']:.3f}/{row['t1']['acc@25km']:.3f}%")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
