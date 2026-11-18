"""Behavioral tests for date encoding and feature/metadata alignment."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest


def test_date_intervals_preserve_years_and_uncertainty():
    from models.geoclip_time.lib.temporal_data import encode_date_intervals

    frame = pd.DataFrame(
        {
            "date_shot_min": [1900, "1950-07-15", "1800", 2000.0],
            "date_shot_max": [1920, "1950-07-15", "2026", 2020.0],
        }
    )
    times, audit = encode_date_intervals(frame)

    np.testing.assert_allclose(
        times,
        [[-0.4, 0.1, 1.0], [0.0, 0.0, 1.0], [-0.37, 1.13, 1.0], [0.6, 0.1, 1.0]],
    )
    assert times.dtype == np.float32
    assert audit["valid"] == 4
    assert audit["invalid"] == 0
    json.dumps(audit, allow_nan=False)


def test_unknown_and_invalid_intervals_are_masked_with_reasons():
    from models.geoclip_time.lib.temporal_data import encode_date_intervals

    frame = pd.DataFrame(
        {
            "date_shot_min": [None, "1900", "garbage", "1950", "1799", "1900", "2027"],
            "date_shot_max": [None, None, "1910", "1940", "1900", "2027", "2028"],
        }
    )
    times, audit = encode_date_intervals(frame)

    np.testing.assert_array_equal(times, np.zeros((7, 3), dtype=np.float32))
    assert audit["valid"] == 0
    assert audit["invalid"] == 7
    assert audit["reasons"] == {
        "missing": 1,
        "one_sided": 1,
        "malformed": 1,
        "reversed": 1,
        "out_of_range": 3,
    }
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize(
    "lower,upper",
    [
        ("1900-12-31", "1900-01-01"),
        (date(1900, 12, 31), date(1900, 1, 1)),
        ("1900-06-01T12:01:00Z", "1900-06-01T12:00:00Z"),
    ],
)
def test_reversed_full_dates_within_one_year_are_unknown(lower, upper):
    from models.geoclip_time.lib.temporal_data import encode_date_intervals

    frame = pd.DataFrame({"date_shot_min": [lower], "date_shot_max": [upper]})
    times, audit = encode_date_intervals(frame)

    np.testing.assert_array_equal(times, [[0, 0, 0]])
    assert audit["reasons"]["reversed"] == 1


def test_timezone_ordering_preserves_recorded_calendar_years():
    from models.geoclip_time.lib.temporal_data import encode_date_intervals

    frame = pd.DataFrame(
        {
            "date_shot_min": ["1900-06-01T00:30:00+02:00", "1900-05-31T23:00:00Z", "1900-01-01T00:00:00+02:00"],
            "date_shot_max": ["1900-05-31T23:00:00Z", "1900-06-01T00:30:00+02:00", "1900-01-01T00:00:00+02:00"],
        }
    )
    times, audit = encode_date_intervals(frame)

    np.testing.assert_allclose(times, [[-0.5, 0, 1], [0, 0, 0], [-0.5, 0, 1]])
    assert audit["reasons"]["reversed"] == 1


def test_full_dates_with_mixed_timezone_awareness_are_unknown():
    from models.geoclip_time.lib.temporal_data import encode_date_intervals

    frame = pd.DataFrame(
        {"date_shot_min": ["1900-01-01T00:00:00Z"], "date_shot_max": ["1900-12-31"]}
    )
    times, audit = encode_date_intervals(frame)

    np.testing.assert_array_equal(times, [[0, 0, 0]])
    assert audit["reasons"]["malformed"] == 1


def test_year_endpoints_do_not_imply_january_first_when_checking_order():
    from models.geoclip_time.lib.temporal_data import encode_date_intervals

    frame = pd.DataFrame(
        {"date_shot_min": ["1900-12-31", "1900-12-31"], "date_shot_max": [1900, "1900"]}
    )
    times, audit = encode_date_intervals(frame)

    np.testing.assert_allclose(times, [[-0.5, 0, 1], [-0.5, 0, 1]])
    assert audit["valid"] == 2


def test_custom_time_scale_and_empty_frame():
    from models.geoclip_time.lib.temporal_data import encode_date_intervals

    frame = pd.DataFrame({"date_shot_min": [1850], "date_shot_max": [1950]})
    times, _ = encode_date_intervals(frame, year_center=1900, year_scale=50)
    np.testing.assert_array_equal(times, [[0, 1, 1]])
    empty, audit = encode_date_intervals(frame.iloc[:0])
    assert empty.shape == (0, 3)
    assert audit["total"] == 0


@pytest.mark.parametrize(
    "config",
    [{"year_scale": 0}, {"year_scale": -1}, {"year_center": np.inf}, {"min_year": 2000, "max_year": 1900}],
)
def test_invalid_time_configuration_fails(config):
    from models.geoclip_time.lib.temporal_data import encode_date_intervals

    frame = pd.DataFrame({"date_shot_min": [1900], "date_shot_max": [1920]})
    with pytest.raises(ValueError):
        encode_date_intervals(frame, **config)


@pytest.fixture
def temporal_files(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "features").mkdir()
    meta = pd.DataFrame(
        {
            "id": [20, 30, 10, 40],
            "latitude": [46.2, 46.3, 46.1, 46.4],
            "longitude": [7.2, 7.3, 7.1, 7.4],
            "view_type": ["nadir", "terrestrial", "lowOblique", "terrestrial"],
            "date_shot_min": ["1950", "1980", "1900", None],
            "date_shot_max": ["1950", None, "1920", None],
        }
    )
    meta.to_parquet(tmp_path / "meta/smapshot.parquet", index=False)
    splits = pd.DataFrame({"id": [10, 40, 30, 20], "split": ["train", "train", "val", "buffer"], "block": [11, 44, 33, 22]})
    split_csv = tmp_path / "split.csv"
    splits.to_csv(split_csv, index=False)
    features = np.zeros((3, 768), dtype=np.float16)
    features[:, 0] = [3, 1, 2]
    np.save(tmp_path / "features/clip_vitl14.npy", features)
    np.save(tmp_path / "features/clip_vitl14_ids.npy", np.array([30, 10, 20]))
    return tmp_path, split_csv


def test_shuffled_ids_retain_every_cached_row_and_correct_associations(temporal_files):
    from models.geoclip_time.lib.temporal_data import load_temporal_data

    root, splits = temporal_files
    data = load_temporal_data(root, splits)

    assert isinstance(data.features, np.memmap)
    assert not data.features.flags.writeable
    assert data.table.id.tolist() == [30, 10, 20]
    assert data.table.feat_idx.tolist() == [0, 1, 2]
    assert data.table.split.tolist() == ["val", "train", "buffer"]
    assert data.table.block.tolist() == [33, 11, 22]
    assert data.table.view_type.tolist() == ["terrestrial", "lowOblique", "nadir"]
    np.testing.assert_allclose(data.table[["latitude", "longitude"]], [[46.3, 7.3], [46.1, 7.1], [46.2, 7.2]])
    np.testing.assert_array_equal(data.features[data.table.feat_idx, 0], [3, 1, 2])
    np.testing.assert_allclose(data.times, [[0, 0, 0], [-0.4, 0.1, 1], [0, 0, 1]])
    assert data.table.date_shot_min.tolist() == ["1980", "1900", "1950"]
    assert data.audit["coverage"] == {
        "train": {"expected": 2, "available": 1, "missing": 1},
        "val": {"expected": 1, "available": 1, "missing": 0},
        "buffer": {"expected": 1, "available": 1, "missing": 0},
    }
    assert data.audit["dates"]["reasons"]["one_sided"] == 1
    json.dumps(data.audit, allow_nan=False)


def test_loader_applies_time_configuration(temporal_files):
    from models.geoclip_time.lib.temporal_data import load_temporal_data

    root, splits = temporal_files
    data = load_temporal_data(root, splits, time_config={"year_center": 1900, "year_scale": 10})
    np.testing.assert_array_equal(data.times[1], [1, 1, 1])


@pytest.mark.parametrize("source", ["metadata", "splits", "features"])
@pytest.mark.parametrize("bad_ids", [[30, 30, 20], [30, None, 20], [30, 10.5, 20]])
def test_corrupt_ids_fail_before_joining(temporal_files, source, bad_ids):
    from models.geoclip_time.lib.temporal_data import load_temporal_data

    root, splits = temporal_files
    if source == "metadata":
        path = root / "meta/smapshot.parquet"
        frame = pd.read_parquet(path).iloc[:3].copy()
        frame["id"] = bad_ids
        frame.to_parquet(path, index=False)
    elif source == "splits":
        frame = pd.read_csv(splits).iloc[:3].copy()
        frame["id"] = bad_ids
        frame.to_csv(splits, index=False)
    else:
        np.save(root / "features/clip_vitl14_ids.npy", np.asarray(bad_ids, dtype=float))
    with pytest.raises(ValueError, match="(?i)id"):
        load_temporal_data(root, splits)


@pytest.mark.parametrize("shape", [(2, 768), (3, 512), (3, 768, 1)])
def test_invalid_feature_dimensions_or_lengths_fail(temporal_files, shape):
    from models.geoclip_time.lib.temporal_data import load_temporal_data

    root, splits = temporal_files
    np.save(root / "features/clip_vitl14.npy", np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match="(?i)(feature|cache)"):
        load_temporal_data(root, splits)


@pytest.mark.parametrize("source", ["metadata", "splits"])
def test_cached_ids_without_metadata_or_split_fail(temporal_files, source):
    from models.geoclip_time.lib.temporal_data import load_temporal_data

    root, splits = temporal_files
    if source == "metadata":
        path = root / "meta/smapshot.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame.id != 30].to_parquet(path, index=False)
    else:
        frame = pd.read_csv(splits)
        frame.loc[frame.id != 30].to_csv(splits, index=False)
    with pytest.raises(ValueError, match="(?i)(missing|without).*(30|id)"):
        load_temporal_data(root, splits)


@pytest.mark.parametrize("column,value", [("latitude", np.nan), ("latitude", 90.1), ("longitude", np.inf), ("longitude", -180.1)])
def test_cached_invalid_coordinates_fail(temporal_files, column, value):
    from models.geoclip_time.lib.temporal_data import load_temporal_data

    root, splits = temporal_files
    path = root / "meta/smapshot.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame.id == 30, column] = value
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="(?i)(coordinate|latitude|longitude)"):
        load_temporal_data(root, splits)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_cached_features_fail(temporal_files, bad_value):
    from models.geoclip_time.lib.temporal_data import load_temporal_data

    root, splits = temporal_files
    path = root / "features/clip_vitl14.npy"
    features = np.load(path)
    features[-1, -1] = bad_value
    np.save(path, features)
    with pytest.raises(ValueError, match="(?i)(finite|NaN|infinity)"):
        load_temporal_data(root, splits)


@pytest.mark.parametrize("column", ["date_shot_min", "date_shot_max"])
def test_missing_date_schema_fails_clearly(temporal_files, column):
    from models.geoclip_time.lib.temporal_data import load_temporal_data

    root, splits = temporal_files
    path = root / "meta/smapshot.parquet"
    pd.read_parquet(path).drop(columns=column).to_parquet(path, index=False)
    with pytest.raises(ValueError, match=column):
        load_temporal_data(root, splits)
