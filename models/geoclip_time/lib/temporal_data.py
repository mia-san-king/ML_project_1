"""Validated, explicitly located inputs for time-conditioned GeoCLIP.

This module does not import ``lib.paths`` or create directories. Feature arrays
stay read-only and memory-mapped; table row i and times row i describe feature
row i, including rows whose split is excluded from training.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class TemporalData:
    features: np.memmap
    table: pd.DataFrame
    times: np.ndarray
    audit: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {', '.join(missing)}")


def _missing_date(value: Any) -> bool:
    return bool(pd.isna(value)) or (isinstance(value, str) and not value.strip())


def _date_endpoint(value: Any) -> tuple[int | None, pd.Timestamp | None]:
    """Preserve date precision without turning a bare year into January 1."""
    if isinstance(value, (bool, np.bool_)):
        return None, None
    if isinstance(value, Real):
        year = int(value) if np.isfinite(value) and float(value).is_integer() else None
        return year, None
    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r"[+-]?\d{4}", value):
            return int(value), None
        # Require a year-first ISO date rather than guessing ambiguous formats.
        if not re.match(r"^\d{4}-\d{2}(?:-\d{2})?(?:$|[T ])", value):
            return None, None
    elif not isinstance(value, (date, datetime, pd.Timestamp, np.datetime64)):
        return None, None
    try:
        timestamp = pd.Timestamp(value)
        return timestamp.year, timestamp
    except (TypeError, ValueError, OverflowError):
        return None, None


def encode_date_intervals(
    frame: pd.DataFrame,
    *,
    year_center: float = 1950.0,
    year_scale: float = 100.0,
    min_year: int = 1800,
    max_year: int = 2026,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode [minimum year, maximum year] as midpoint, half-width and mask.

    The first two columns are divided by ``year_scale`` and the midpoint is
    centered on ``year_center``. Numeric calendar years and ISO dates are
    accepted. Both endpoints must be present and valid: no exact year is
    inferred from one endpoint. Missing, one-sided, malformed, reversed and
    out-of-range intervals become [0, 0, 0], with exclusive counts in ``reasons``.
    Bounds are inclusive. Full dates are also checked for chronological order
    before reducing to years: timezone-aware endpoints compare as instants,
    naive endpoints as written. Mixed aware/naive full dates are malformed.
    Calendar years always retain each endpoint's recorded timezone, and bare
    years do not imply January 1. Subyear precision is not a model feature.
    """
    _require_columns(frame, ["date_shot_min", "date_shot_max"], "metadata")
    config = {
        "year_center": float(year_center),
        "year_scale": float(year_scale),
        "min_year": float(min_year),
        "max_year": float(max_year),
    }
    if not all(np.isfinite(value) for value in config.values()):
        raise ValueError("time configuration values must be finite")
    if year_scale <= 0 or min_year > max_year:
        raise ValueError("year_scale must be positive and min_year must not exceed max_year")

    times = np.zeros((len(frame), 3), dtype=np.float32)
    reasons = dict.fromkeys(["missing", "one_sided", "malformed", "reversed", "out_of_range"], 0)
    for index, (lower, upper) in enumerate(
        frame[["date_shot_min", "date_shot_max"]].itertuples(index=False, name=None)
    ):
        lower_missing, upper_missing = _missing_date(lower), _missing_date(upper)
        if lower_missing or upper_missing:
            reason = "missing" if lower_missing and upper_missing else "one_sided"
        else:
            lower_year, lower_date = _date_endpoint(lower)
            upper_year, upper_date = _date_endpoint(upper)
            both_dates = lower_date is not None and upper_date is not None
            if lower_year is None or upper_year is None:
                reason = "malformed"
            elif both_dates and (lower_date.tzinfo is None) != (upper_date.tzinfo is None):
                reason = "malformed"
            elif lower_year > upper_year or (both_dates and lower_date > upper_date):
                reason = "reversed"
            elif lower_year < min_year or upper_year > max_year:
                reason = "out_of_range"
            else:
                times[index] = (
                    ((lower_year + upper_year) / 2.0 - year_center) / year_scale,
                    (upper_year - lower_year) / (2.0 * year_scale),
                    1.0,
                )
                continue
        reasons[reason] += 1
    invalid = sum(reasons.values())
    return times, {
        "total": len(frame),
        "valid": len(frame) - invalid,
        "invalid": invalid,
        "reasons": reasons,
        "config": config,
    }


def _validated_ids(values: Any, source: str) -> np.ndarray:
    """Validate before coercion so nulls, fractions and overflows cannot join."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{source} IDs must be a one-dimensional array")
    result = np.empty(len(array), dtype=np.int64)
    limits = np.iinfo(np.int64)
    for index, value in enumerate(array):
        if pd.isna(value):
            raise ValueError(f"{source} contains a null ID at row {index}")
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{source} contains a noninteger ID at row {index}")
        if isinstance(value, Integral):
            image_id = int(value)
        elif isinstance(value, Real) and np.isfinite(value) and float(value).is_integer():
            image_id = int(value)
        elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            image_id = int(value)
        else:
            raise ValueError(f"{source} contains a noninteger ID at row {index}: {value!r}")
        if not limits.min <= image_id <= limits.max:
            raise ValueError(f"{source} ID outside int64 range at row {index}")
        result[index] = image_id
    duplicate = pd.Series(result).duplicated()
    if duplicate.any():
        examples = result[duplicate.to_numpy()][:5].tolist()
        raise ValueError(f"{source} contains duplicate IDs: {examples}")
    return result


def load_temporal_data(
    data_root: str | Path,
    split_csv: str | Path,
    *,
    feature_name: str = "clip_vitl14",
    time_config: Mapping[str, Any] | None = None,
) -> TemporalData:
    """Join cached features, metadata and splits without dropping cached rows.

    Reads ``meta/smapshot.parquet``, ``features/{feature_name}.npy`` and
    ``features/{feature_name}_ids.npy`` under ``data_root``.
    ``split_csv`` is an explicit path, independent of the data root. Metadata
    requires id, latitude, longitude and both date_shot_min/date_shot_max columns;
    the split CSV requires id and split. Absent optional view_type/block columns
    are returned as nulls. Every source must have unique, non-null integer IDs.

    All cached IDs must have metadata and a split. Splits without cached images
    remain visible in audit["coverage"], whose expected/available/missing counts
    let callers enforce completeness for the particular splits they use. No
    train/validation/test filtering or completeness policy is applied here.
    """
    root = Path(data_root).expanduser()
    metadata_path = root / "meta" / "smapshot.parquet"
    required = ["id", "latitude", "longitude", "date_shot_min", "date_shot_max"]
    metadata_columns = pq.read_schema(metadata_path).names
    missing = sorted(set(required).difference(metadata_columns))
    if missing:
        raise ValueError(f"metadata missing required columns: {', '.join(missing)}")
    columns = required + (["view_type"] if "view_type" in metadata_columns else [])
    metadata = pd.read_parquet(metadata_path, columns=columns)
    splits = pd.read_csv(Path(split_csv).expanduser(), dtype={"id": "string"})
    _require_columns(splits, ["id", "split"], "split CSV")
    metadata["id"] = _validated_ids(metadata["id"], "metadata")
    splits["id"] = _validated_ids(splits["id"], "split CSV")
    if splits["split"].isna().any() or splits["split"].astype(str).str.strip().eq("").any():
        raise ValueError("split CSV contains a missing split label")
    splits["split"] = splits["split"].astype(str)
    if "view_type" not in metadata:
        metadata["view_type"] = None
    if "block" not in splits:
        splits["block"] = None

    features = np.load(root / "features" / f"{feature_name}.npy", mmap_mode="r", allow_pickle=False)
    ids = _validated_ids(
        np.load(root / "features" / f"{feature_name}_ids.npy", allow_pickle=False),
        "feature cache",
    )
    if features.ndim != 2 or features.shape[1] != 768:
        raise ValueError(f"feature cache must have shape (N, 768); got {features.shape}")
    if features.shape[0] != len(ids):
        raise ValueError(f"feature cache length {len(features)} differs from ID count {len(ids)}")
    if not np.issubdtype(features.dtype, np.number) or np.issubdtype(features.dtype, np.complexfloating):
        raise ValueError("feature cache must contain real numeric features")
    # Bound the temporary finite mask rather than allocating a full-cache copy.
    for start in range(0, len(features), 8192):
        if not np.isfinite(features[start : start + 8192]).all():
            raise ValueError(f"feature cache contains nonfinite values near row {start}")

    for source, frame in (("metadata", metadata), ("split CSV", splits)):
        absent = ids[~np.isin(ids, frame["id"].to_numpy())]
        if len(absent):
            raise ValueError(f"cached IDs missing from {source}: {absent[:5].tolist()} ({len(absent)} total)")

    table = pd.DataFrame({"feat_idx": np.arange(len(ids), dtype=np.int64), "id": ids})
    table = table.merge(metadata, on="id", how="left", validate="one_to_one", sort=False)
    table = table.merge(splits[["id", "split", "block"]], on="id", how="left", validate="one_to_one", sort=False)
    table = table.sort_values("feat_idx").reset_index(drop=True)
    coordinates = table[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    valid_coordinates = (
        np.isfinite(coordinates).all(axis=1)
        & (np.abs(coordinates[:, 0]) <= 90)
        & (np.abs(coordinates[:, 1]) <= 180)
    )
    if not valid_coordinates.all():
        bad_ids = table.loc[~valid_coordinates, "id"].head().tolist()
        raise ValueError(f"invalid latitude/longitude coordinates for cached IDs: {bad_ids}")
    table["latitude"], table["longitude"] = coordinates[:, 0], coordinates[:, 1]
    times, date_audit = encode_date_intervals(table, **dict(time_config or {}))
    available = table["split"].value_counts()
    coverage = {}
    for split, expected in splits["split"].value_counts().items():
        count = int(available.get(split, 0))
        coverage[split] = {"expected": int(expected), "available": count, "missing": int(expected) - count}
    audit = {
        "coverage": coverage,
        "dates": date_audit,
        "feature_rows": len(features),
        "feature_dim": int(features.shape[1]),
        "metadata_rows": len(metadata),
        "split_rows": len(splits),
    }
    return TemporalData(features=features, table=table, times=times, audit=audit)
