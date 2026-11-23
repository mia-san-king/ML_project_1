import json

import numpy as np
import pandas as pd
import pytest

from models.t6_router.expanded_regions import SEVEN
from models.t7_error_regression import model, sample_size


def samples():
    rows = []
    for i, name in enumerate(SEVEN):
        for j in range(5 if i == 0 else 3):
            rows.append({"id": 10*i+j+1, "region": name, "block": i, "T4_lat_pred": 46., "T4_lon_pred": 7.,
                         "T5_lat_pred": 46.5, "T5_lon_pred": 8., "has_date": True, "t4_was_projected": False,
                         "nearest_train_km": 1., "log1p_train_5km": .6, "log1p_train_25km": 2., "t5_visual_rank": 1,
                         "t5_t3_score": .4, "t5_location_cosine": np.nan, "T4_error_km": [8., 12., 4., 5., 3.][j],
                         "T5_error_km": [10., 5., 4., 5., 8.][j]})
    full = pd.DataFrame(rows)
    return full.loc[~full.id.isin([4, 5])].copy(), full


def test_both_arms_hold_out_full_region_and_counts_reconcile(monkeypatch):
    matched, full = samples()
    calls = []

    def fit(source):
        assert set(source.columns) == {*model.INPUT_COLUMNS, "T4_error_km", "T5_error_km"}
        return set(source.index)

    def predict(trained, queries):
        assert tuple(queries.columns) == model.INPUT_COLUMNS
        assert trained.isdisjoint(queries.index)
        calls.append((trained, set(queries.index)))
        choose4 = 4 in trained
        return pd.DataFrame({"T4_expected_km": 3. if choose4 else 9., "T5_expected_km": 6.,
                             "T7_choose_T4": choose4}, index=queries.index)

    monkeypatch.setattr(model, "fit", fit)
    monkeypatch.setattr(model, "predict", predict)
    result = sample_size.evaluate_sample_size(matched, full)
    assert len(calls) == 14
    for name in SEVEN:
        train_sets = [source for source, held in calls if held == set(full.loc[full.region.eq(name), "id"])]
        assert len(train_sets) == 2
        assert set(matched.loc[matched.region.ne(name), "id"]) in train_sets
        assert set(full.loc[full.region.ne(name), "id"]) in train_sets
    assert result["folds"]["dev1"]["same_training_ids"]
    assert result["folds"]["dev1"]["identical_forecasts"] and result["folds"]["dev1"]["identical_choices"]
    assert result["affected_regions"] == list(SEVEN[1:])
    assert len(full) == result["pooled"]["n"] == 23
    assert result["matched_evaluation"]["n"] == 21
    assert result["affected_evaluation"]["n"] == 18
    pair = result["pooled"]["paired"]
    assert (pair["closer"], pair["farther"], pair["unchanged"], pair["switched"], pair["net_closer"]) == (6, 6, 11, 18, 0)
    assert pair["total_saved_km"] == -30 and pair["mean_saved_km"] == pytest.approx(-30/23)
    assert result["matched_evaluation"]["paired"]["mean_saved_km"] == pytest.approx(-30/21)
    assert result["affected_evaluation"]["paired"]["mean_saved_km"] == pytest.approx(-30/18)
    assert result["macro"]["paired"]["mean_saved_km"] == pytest.approx(-10/7)
    for key, rows in (("pooled", full), ("matched_evaluation", matched), ("affected_evaluation", full[full.region.ne("dev1")])):
        summary = result[key]
        for arm in ("matched", "full"):
            select4 = (arm == "full") & rows.region.ne("dev1")
            chosen = np.where(select4, rows.T4_error_km, rows.T5_error_km)
            non_ties = rows.T4_error_km.ne(rows.T5_error_km).to_numpy()
            closer = (chosen == np.minimum(rows.T4_error_km, rows.T5_error_km).to_numpy()) & non_ties
            delta4 = np.where(select4, 3., 9.) - rows.T4_error_km
            delta5 = 6. - rows.T5_error_km
            assert summary[arm]["closer_accuracy"] == pytest.approx(closer.sum() / non_ties.sum())
            assert summary[arm]["chosen_mean_km"] == pytest.approx(chosen.mean())
            assert summary[arm]["chosen_median_km"] == pytest.approx(np.median(chosen))
            assert summary[arm]["forecast_joint_rmse_km"] == pytest.approx(np.sqrt(np.mean((delta4**2 + delta5**2) / 2)))
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_identical_samples_reproduce_original_t7_no_matter_input_order():
    _, full = samples()
    result = sample_size.evaluate_sample_size(full.sample(frac=1, random_state=12), full.iloc[::-1])
    assert result["parameters"] == model.REGRESSOR_PARAMS and result["parameters"]["max_iter"] == 80
    assert result["affected_regions"] == [] and result["affected_evaluation"] is None
    assert result["matched_evaluation"] == result["pooled"]
    assert result["pooled"]["paired"]["unchanged"] == len(full)
    assert result["pooled"]["paired"]["switched"] == 0
    true = full.loc[full.region.eq("dev3")].sort_values("id")
    predicted = model.predict(model.fit(full.loc[full.region.ne("dev3")].sort_values("id")), true)
    chosen = np.where(predicted.T7_choose_T4, true.T4_error_km, true.T5_error_km)
    assert result["by_region"]["dev3"]["full"]["chosen_mean_km"] == pytest.approx(chosen.mean())
    assert result["by_region"]["dev3"]["full"] == result["by_region"]["dev3"]["matched"]


def test_rejects_non_nested_or_changed_shared_examples_and_invalid_regions():
    matched, full = samples()
    for invalid in (matched.assign(id=matched.id+1000), matched.assign(t5_t3_score=.41),
                    matched.assign(T4_error_km=42.), pd.concat([matched, matched.iloc[:1]]),
                    matched.assign(region="new_final"), matched.assign(block=0)):
        with pytest.raises(ValueError):
            sample_size.evaluate_sample_size(invalid, full)
