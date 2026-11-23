import json

import numpy as np
import pandas as pd
import pytest

from models.t6_router.expanded_regions import SEVEN
from models.t7_error_regression import learning_curve as curve, model


def sample():
    region = np.repeat(SEVEN, np.arange(47, 54))
    ids = np.arange(len(region)) + 1
    easy = ids % 3 == 0
    return pd.DataFrame({
        "region": region, "block": region, "id": ids, "T4_lat_pred": 46.4, "T4_lon_pred": 6.3,
        "T5_lat_pred": np.where(easy, 47.4, 46.4), "T5_lon_pred": 8.4, "has_date": True,
        "t4_was_projected": False, "nearest_train_km": 1., "log1p_train_5km": .69,
        "log1p_train_25km": 3., "t5_visual_rank": ids, "t5_t3_score": .8, "t5_location_cosine": .1,
        "T4_error_km": np.where(easy, 6., 45.) + ids % 7,
        "T5_error_km": np.where(easy, 62., 10.) + ids % 7,
    })


def test_80_trees_reproduces_original_t7_in_each_fold_and_pooling():
    data = sample().sample(frac=1., random_state=71)
    # This region tests that undefined accuracy does not turn into a JSON NaN or a spurious zero.
    tie_region = data.region.eq(SEVEN[-1])
    data.loc[tie_region, "T4_error_km"] = data.loc[tie_region, "T5_error_km"]
    report = curve.evaluate_curve(data, stages=(4, 80), fractions=(1.,))
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    point = report["points"][-1]
    assert point["fraction"] == 1. and point["stage"] == 80
    estimates, truth, chosen_errors, train_sse, train_n = [], [], [], 0., 0
    for name in SEVEN:
        train = data.loc[data.region.ne(name)].sort_values("id")
        held = data.loc[data.region.eq(name)].sort_values("id")
        baseline = model.fit(train)
        history = model.predict(baseline, train)[["T4_expected_km", "T5_expected_km"]].to_numpy()
        expected = model.predict(baseline, held)
        targets = held[["T4_error_km", "T5_error_km"]].to_numpy()
        forecasts = expected[["T4_expected_km", "T5_expected_km"]].to_numpy()
        selected = np.where(expected.T7_choose_T4, targets[:, 0], targets[:, 1])
        train_delta = history - train[["T4_error_km", "T5_error_km"]].to_numpy()
        fold = point["by_region"][name]
        assert fold["train_joint_rmse_km"] == pytest.approx(np.sqrt(np.mean(train_delta**2)))
        assert fold["held_joint_rmse_km"] == pytest.approx(np.sqrt(np.mean((forecasts - targets)**2)))
        assert fold["chosen_median_km"] == pytest.approx(np.median(selected))
        train_n += len(train)
        train_sse += np.sum(train_delta**2)
        estimates.extend(forecasts)
        truth.extend(targets)
        chosen_errors.extend(selected)
    delta = np.array(estimates) - np.array(truth)
    pooled = point["pooled"]
    assert pooled["held_n"] == len(data) and pooled["train_n"] == train_n
    assert pooled["train_joint_rmse_km"] == pytest.approx(np.sqrt(train_sse / (2 * train_n)))
    assert pooled["held_joint_rmse_km"] == pytest.approx(np.sqrt(np.mean(delta**2)))
    assert pooled["held_T4_rmse_km"] == pytest.approx(np.sqrt(np.mean(delta[:, 0]**2)))
    assert pooled["held_T5_mae_km"] == pytest.approx(np.mean(abs(delta[:, 1])))
    assert pooled["chosen_median_km"] == pytest.approx(np.median(chosen_errors))
    assert pooled["non_ties"] == len(data) - tie_region.sum()
    assert pooled["closer_correct"] == np.count_nonzero((np.min(truth, axis=1) == chosen_errors)
                                                       & (np.asarray(truth)[:, 0] != np.asarray(truth)[:, 1]))
    assert point["by_region"][SEVEN[-1]]["closer_accuracy"] is None
    assert point["macro"]["closer_accuracy"] == pytest.approx(np.mean(
        [point["by_region"][name]["closer_accuracy"] for name in SEVEN[:-1]]))
    assert point["macro"]["chosen_median_km"] == pytest.approx(np.mean(
        [fold["chosen_median_km"] for fold in point["by_region"].values()]))


def test_regions_are_disjoint_both_targets_share_nested_rows_and_tie_goes_to_t5(monkeypatch):
    data = sample()
    data["T4_error_km"] = 1.
    data["T5_error_km"] = 8.
    seen = []

    class Spy:
        def __init__(self, **kwargs):
            assert kwargs == {**model.REGRESSOR_PARAMS, "max_iter": 4}

        def fit(self, x, y):
            assert tuple(x.columns) == model.FEATURE_COLUMNS
            ids = x.t5_visual_rank.astype(int).to_numpy()
            seen.append((frozenset(ids), float(y[0])))
            return self

        def staged_predict(self, x):
            for _ in range(4):
                yield np.full(len(x), -2.)  # Both errors clip to zero, so T5 must win.

    monkeypatch.setattr(curve, "HistGradientBoostingRegressor", Spy)
    report = curve.evaluate_curve(data, stages=(1, 4), fractions=(.5, 1.), seed=18)
    assert len(seen) == 7 * 2 * 2
    counts = data.region.value_counts()
    groups = {name: set(data.loc[data.region.eq(name), "id"]) for name in SEVEN}
    for name, held in groups.items():
        fits = [(source, y) for source, y in seen if held.isdisjoint(source)]
        assert len(fits) == 4
        samples = sorted({source for source, _ in fits}, key=len)
        half, full = samples
        assert half < full and full == set(data.id) - held
        fold = report["folds"][name]
        assert fold["held_n"] == len(held)
        for frac_key, ids in (("0.5", half), ("1.0", full)):
            assert {y for source, y in fits if source == ids} == {1., 8.}
            counts_seen = data.loc[data.id.isin(ids)].region.value_counts().to_dict()
            allocated = fold["train_by_fraction"][frac_key]
            assert allocated["train_n"] == len(ids) and allocated["photos_by_region"] == counts_seen
            assert all(n == int(counts[other] * float(frac_key)) for other, n in counts_seen.items())
    for point in report["points"]:
        assert point["pooled"]["chosen_mean_km"] == 8
        assert point["pooled"]["held_T4_rmse_km"] == 1
        assert point["pooled"]["closer_correct"] == 0
    shuffled = data.sample(frac=1., random_state=5)
    assert curve.evaluate_curve(shuffled, stages=(1, 4), fractions=(.5, 1.), seed=18) == report


def test_rejects_invalid_splits_and_curve_settings():
    data = sample()
    for bad in (data.assign(id=1), data.assign(block=1), data.loc[data.region.ne("dev6")]):
        with pytest.raises(ValueError):
            curve.evaluate_curve(bad, stages=(1,))
    for options in ({"stages": ()}, {"stages": (0, 3)}, {"stages": (1.1,)}, {"fractions": (.5, 1.5)}):
        with pytest.raises(ValueError):
            curve.evaluate_curve(data, **options)
