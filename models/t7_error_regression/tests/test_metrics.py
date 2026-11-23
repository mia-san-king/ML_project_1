"""Hand-calculated distance, routing and expected-km calibration diagnostics."""
import json

import numpy as np
import pandas as pd
import pytest


def sample():
    return pd.DataFrame({
        "T4_error_km": [35., 15., 5., 60., 70., 45., 12., 25.],
        "T5_error_km": [80., 3., 41., 6., 45., 75., 12., 25.],
        "T4_expected_km": [0., 10., 25., 50., 100., 0., 10., 25.],
        "T5_expected_km": [20.] * 8,
        "T7_choose_T4": [True, False, False, True, True, True, True, False],
        "T6_choose_T4": [False, True, True, False, True, False, False, True],
    })


def evaluate(frame):
    # Import here so a missing implementation is a failed behavioral test,
    # rather than preventing pytest from collecting all the test cases.
    from models.t7_error_regression.metrics import evaluate as implementation
    return implementation(frame)


def test_reports_expert_and_router_distances_with_exact_ties_excluded_from_closer_rate():
    frame = sample()
    original = frame.copy(deep=True)
    result = evaluate(frame)
    assert result["n"] == 8
    expected = {
        "T7": (38., 291 / 8, 3, 50., 3, 5),
        "T6": (20., 36., 2, 100 / 3, 5, 4),
        "always_T4": (30., 267 / 8, 3, 50., 4, 8),
        "always_T5": (33., 287 / 8, 3, 50., 4, 0),
    }
    for name, (median, mean, correct, closer_pct, hits, choose4) in expected.items():
        metrics = result["methods"][name]
        assert metrics["n_non_ties"] == 6 and metrics["n_ties"] == 2
        assert metrics["median_km"] == pytest.approx(median)
        assert metrics["mean_km"] == pytest.approx(mean)
        assert metrics["correct_non_ties"] == correct
        assert metrics["accuracy_non_ties_pct"] == pytest.approx(closer_pct)
        assert metrics["hits_25km"] == hits
        assert metrics["hit_25km_pct"] == pytest.approx(hits * 12.5)
        assert metrics["choose_T4"] == choose4
    pd.testing.assert_frame_equal(frame, original)


def test_reports_paired_switches_even_when_closer_accuracy_and_mean_error_disagree():
    change = evaluate(sample())["paired_vs_T6"]
    for key, value in {
        "switches": 7, "T5_to_T4": 4, "T4_to_T5": 3,
        "improved": 3, "worsened": 2, "equal_after_switch": 2,
        "saved_km": 87, "added_km": 90, "net_saved_km": -3,
        "mean_saved_km": -3 / 8, "net_correct_non_ties": 1,
    }.items():
        assert change[key] == pytest.approx(value)
    assert change["hit25_changes"] == {
        "additional_T4_only_rescues": 0, "previous_T4_only_rescues_reversed": 1,
        "additional_T5_only_losses": 1, "previous_T5_only_losses_recovered": 0,
        "net_T4_only_rescues": -1, "net_T5_only_losses": 1, "net_hits_25km": -2,
    }


def test_expected_error_mae_and_calibration_use_predicted_km_with_left_closed_bins():
    result = evaluate(sample())["expected_km"]
    t4 = result["T4"]
    assert t4["mae_km"] == pytest.approx(147 / 8)
    assert t4["mean_predicted_km"] == 27.5
    assert t4["mean_observed_km"] == pytest.approx(267 / 8)
    bins = t4["calibration_bins"]
    assert [row["bin"] for row in bins] == ["0-10", "10-25", "25-50", "50-100", "100+"]
    assert [row["count"] for row in bins] == [2, 2, 2, 1, 1]
    assert [row["mean_predicted_km"] for row in bins] == [0., 10., 25., 50., 100.]
    assert [row["mean_observed_km"] for row in bins] == [40., 13.5, 15., 60., 70.]
    assert result["T5"]["mae_km"] == pytest.approx(205 / 8)
    assert result["T5"]["mean_predicted_km"] == 20.
    assert result["T5"]["mean_observed_km"] == pytest.approx(287 / 8)


def test_all_ties_and_empty_calibration_bins_remain_strict_json_ready():
    result = evaluate(sample().assign(T4_error_km=25., T5_error_km=25.))
    for metrics in result["methods"].values():
        assert metrics["n_non_ties"] == 0 and metrics["n_ties"] == 8
        assert metrics["correct_non_ties"] == 0
        assert metrics["accuracy_non_ties_pct"] is None
        assert metrics["hits_25km"] == 8  # The threshold includes exactly 25 km.
    for index in [0, 2, 3, 4]:
        row = result["expected_km"]["T5"]["calibration_bins"][index]
        assert row["count"] == 0
        assert row["mean_predicted_km"] is None
        assert row["mean_observed_km"] is None
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("column,value", [
    ("T4_error_km", -1.), ("T5_error_km", np.nan),
    ("T4_expected_km", -1.), ("T4_expected_km", np.inf),
    ("T5_expected_km", np.nan), ("T7_choose_T4", "False"),
    ("T6_choose_T4", None),
])
def test_rejects_invalid_distances_predictions_and_nonboolean_decisions(column, value):
    frame = sample()
    frame[column] = value
    with pytest.raises(ValueError):
        evaluate(frame)


def test_rejects_empty_missing_or_ambiguous_input_columns():
    frame = sample()
    for invalid in [frame.iloc[:0], frame.drop(columns="T4_expected_km"),
                    pd.concat([frame, frame[["T5_error_km"]]], axis=1)]:
        with pytest.raises(ValueError):
            evaluate(invalid)
