import joblib
import numpy as np
import pandas as pd
import pytest

from models.t7_error_regression import model as t7


def rows(n=400):
    regime = np.arange(n) % 2 == 1
    frame = pd.DataFrame({
        "T4_lat_pred": np.where(regime, 46., 47.), "T4_lon_pred": 7.,
        "T5_lat_pred": 46., "T5_lon_pred": 8.,
        "has_date": regime, "t4_was_projected": False, "log1p_train_5km": 2.,
        "log1p_train_25km": 3., "nearest_train_km": 1.,
        "t5_visual_rank": 1., "t5_t3_score": .5, "t5_location_cosine": .4,
        "T4_error_km": np.where(regime, 5., 90.), "T5_error_km": np.where(regime, 80., 12.),
        "id": np.arange(n) + 1, "region": np.where(regime, "A", "B"), "latitude": 46.5, "longitude": 7.6,
    }, index=pd.Index(np.arange(n) * 17, name="query"))
    return frame


def test_learns_two_error_targets_and_ignores_truth_metadata_and_column_order(tmp_path):
    train = rows()
    train.loc[train.index[::3], "t5_location_cosine"] = np.nan
    fitted = t7.fit(train)
    path = tmp_path / "t7.joblib"
    joblib.dump(fitted, path)
    queries = train.iloc[[3, 1, 2, 0]].copy()
    expected = t7.predict(fitted, queries)
    assert expected.index.equals(queries.index)
    assert list(expected) == ["T4_expected_km", "T5_expected_km", "T7_choose_T4"]
    assert expected.T7_choose_T4.tolist() == [True, True, False, False]
    np.testing.assert_allclose(expected[["T4_expected_km", "T5_expected_km"]],
                               queries[["T4_error_km", "T5_error_km"]], atol=.2)
    queries["expert_gap_km"] = -9999
    queries[["latitude", "longitude", "T4_error_km", "T5_error_km"]] = np.inf
    queries["region"] = "unseen"
    pd.testing.assert_frame_equal(t7.predict(joblib.load(path), queries[queries.columns[::-1]]), expected)
    pd.testing.assert_frame_equal(t7.predict(fitted, queries[list(t7.INPUT_COLUMNS)]), expected)


def test_gap_is_in_kilometres_and_single_row_fit_ties_choose_t5():
    one = rows(1).assign(T4_lat_pred=0., T4_lon_pred=0., T5_lat_pred=0., T5_lon_pred=1.,
                         T4_error_km=33., T5_error_km=33.)
    x = t7._inputs(one)
    assert x["expert_gap_km"].iloc[0] == pytest.approx(111.19508)
    assert tuple(x) == t7.FEATURE_COLUMNS
    assert len(t7.INPUT_COLUMNS) == 12 and len(t7.FEATURE_COLUMNS) == 13
    model = t7.fit(one)
    out = t7.predict(model, one)
    assert out.T4_expected_km.iloc[0] == out.T5_expected_km.iloc[0] == 33.
    assert not out.T7_choose_T4.iloc[0]
    pd.testing.assert_frame_equal(t7.predict(model, one.iloc[:0]), out.iloc[:0])


def test_invalid_coordinates_features_targets_or_nonfinite_model_outputs_fail():
    train = rows(5)
    for column, invalid in (("T4_lat_pred", np.nan), ("T5_lon_pred", 181.), ("t5_t3_score", np.inf),
                            ("T4_error_km", -1.), ("T5_error_km", np.nan), ("T5_error_km", np.inf)):
        with pytest.raises(ValueError):
            t7.fit(train.assign(**{column: invalid}))
    with pytest.raises(ValueError):
        t7.fit(train.iloc[:0])
    with pytest.raises(ValueError):
        t7.fit(pd.concat([train, train[["T4_lat_pred"]]], axis=1))
    queries = train[list(t7.INPUT_COLUMNS)].copy()

    class Dummy:
        def __init__(self, value):
            self.value = value

        def predict(self, x):
            return np.full(len(x), self.value)

    estimates = t7.predict(t7.ExpectedErrorModel(Dummy(-.1), Dummy(3.)), queries)
    assert (estimates.T4_expected_km == 0.).all() and estimates.T7_choose_T4.all()
    with pytest.raises(ValueError, match="finite"):
        t7.predict(t7.ExpectedErrorModel(Dummy(-np.inf), Dummy(3.)), queries)
