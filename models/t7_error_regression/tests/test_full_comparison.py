import json

import numpy as np
import pandas as pd
import pytest

from models.t7_error_regression.full_comparison import compare_decisions


def snapshots():
    rows = pd.DataFrame({
        "id": np.arange(1, 9), "split": ["test_in", "test_city", "test_city", "test_region",
                                        "test_region", "test_in", "new_final", "new_final"],
        "block": [10, 20, 20, 30, 30, 11, 42, 43],
        "T4_lat_pred": 46.1, "T4_lon_pred": 6.7, "T5_lat_pred": 46.8, "T5_lon_pred": 7.5,
        "T4_error_km": [25., 161., 4., 70., 9., 30., 25., 3.],
        "T5_error_km": [30., 7., 5., 50., 9., 29., 25., 3.], "T6_choose_T4": False,
    })
    def decide(picks):
        output = rows.assign(T7_choose_T4=picks, T4_expected_km=np.where(picks, 10., 20.), T5_expected_km=20.)
        for field in ("lat_pred", "lon_pred", "error_km"):
            output["T7_" + field] = np.where(picks, rows["T4_" + field], rows["T5_" + field])
        return output

    return decide([False, False, True, True, False, False, True, False]), decide([True, True, False, False, True, False, False, False])


def test_aligns_on_ids_accounts_for_every_official_change_and_separates_reserved_set():
    old, full = snapshots()
    result = compare_decisions(old.iloc[::-1], full.sample(frac=1, random_state=18))
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    pooled = result["official"]["pooled"]
    assert pooled["n"] == 6
    for key, errors in (("original", [30, 7, 4, 70, 9, 29]), ("full", [25, 161, 5, 50, 9, 29])):
        metrics = pooled[key]
        assert metrics["mean_km"] == pytest.approx(np.mean(errors))
        assert metrics["median_km"] == np.median(errors)
        assert metrics["correct_non_ties"] == 3 and metrics["accuracy_non_ties_pct"] == 60.
        assert metrics["hits_25km"] == 3
    pair = pooled["paired"]
    for key, value in {"switches": 5, "improved": 2, "worsened": 2, "unchanged": 2, "equal_after_switch": 1,
                       "net_correct_non_ties": 0, "net_saved_km": -130, "mean_saved_km": -130 / 6}.items():
        assert pair[key] == pytest.approx(value)
    assert pair["hit25_changes"]["additional_T4_only_rescues"] == 1
    assert pair["hit25_changes"]["additional_T5_only_losses"] == 1
    splits = result["official"]["splits"]
    assert set(splits) == {"test_in", "test_city", "test_region"}
    assert [splits[name]["paired"]["net_correct_non_ties"] for name in ("test_in", "test_city", "test_region")] == [1, -2, 1]
    assert result["new_final"]["n"] == 2
    assert result["new_final"]["full"]["accuracy_non_ties_pct"] is None
    assert result["new_final"]["paired"]["switches"] == 1 and result["new_final"]["paired"]["unchanged"] == 2


@pytest.mark.parametrize(("column", "value"), [
    ("id", 2), ("id", 512), ("split", "test_region"), ("block", 29),
    ("T4_lon_pred", 6.70000000001), ("T5_error_km", 30.000000001),
    ("T7_lat_pred", 46.8), ("T7_lon_pred", 7.5), ("T7_error_km", -4.),
    ("T7_choose_T4", False), ("T4_expected_km", 20.), ("T5_expected_km", np.nan),
])
def test_rejects_different_evaluation_data_and_inconsistent_hard_choices(column, value):
    old, full = snapshots()
    full.loc[0, column] = value
    with pytest.raises(ValueError):
        compare_decisions(old, full)


def test_requires_complete_evaluation_cohorts_and_compares_available_ground_truth():
    old, full = snapshots()
    for subset in (full.iloc[:6], full.iloc[1:]):
        with pytest.raises(ValueError):
            compare_decisions(old, subset)
    with pytest.raises(ValueError):
        compare_decisions(old.assign(latitude=46.2, longitude=6.5), full.assign(latitude=46.3, longitude=6.5))
