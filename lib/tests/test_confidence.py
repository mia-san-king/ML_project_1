"""Contract for per-model, geographically held-out confidence calibration."""

import numpy as np
import pandas as pd
import pytest

from lib.confidence import GeoConfidence
from lib.confidence_features import GalleryFeatureExtractor


def sample_predictions():
    rng = np.random.default_rng(123)
    groups = np.repeat([f"block-{i}" for i in range(12)], 30)
    ids = np.arange(len(groups))
    scores = rng.normal(size=len(groups))
    errors = np.where(scores > 0, 4.0, 60.0)
    features = pd.DataFrame({"native_score": scores, "gallery_count": rng.integers(1, 60, len(groups))})
    return features, errors, groups, ids


def test_geographic_blocks_and_repeated_queries_stay_together():
    features, errors, groups, ids = sample_predictions()
    # Duplicate predictions of several photos must always land in the same partition.
    rows = np.r_[np.arange(len(ids)), np.arange(0, len(ids), 11)]
    model = GeoConfidence(["native_score"]).fit(
        features.iloc[rows], errors[rows], groups=groups[rows], query_ids=ids[rows]
    )
    fit_blocks = set(model.fit_groups_)
    calibration_blocks = set(model.calibration_groups_)
    assert fit_blocks and calibration_blocks
    assert fit_blocks.isdisjoint(calibration_blocks)
    assert fit_blocks | calibration_blocks == set(groups)
    fit_queries = ids[rows][np.isin(groups[rows], list(fit_blocks))]
    calibration_queries = ids[rows][np.isin(groups[rows], list(calibration_blocks))]
    assert set(fit_queries).isdisjoint(calibration_queries)

    conflicting_ids = ids.copy()
    conflicting_ids[-1] = ids[0]
    with pytest.raises(ValueError):
        GeoConfidence(["native_score"]).fit(features, errors, groups=groups, query_ids=conflicting_ids)


def test_one_block_cannot_be_used_for_both_fitting_and_calibration():
    features = pd.DataFrame({"native_score": [0.81, 0.69]})
    with pytest.raises(ValueError):
        GeoConfidence(["native_score"]).fit(features, [2.0, 55.0], groups=["Bern", "Bern"])


@pytest.mark.parametrize("threshold, hit", [(25.0, True), (24.99, False)])
def test_exact_threshold_and_single_class_use_calibration_counts(threshold, hit):
    # Even two observations are enough to exercise the single-class fallback.
    features = pd.DataFrame({"gallery_count": [1, 2]})
    model = GeoConfidence(["gallery_count"], threshold_km=threshold).fit(
        features, [25.0, 25.0], groups=["west", "east"]
    )
    expected = (int(hit) + 1) / 3  # One held-out observation, with Laplace smoothing.
    np.testing.assert_allclose(model.predict_proba(features), [expected, expected])


def test_native_score_is_optional_and_saved_model_can_predict_without_labels(tmp_path):
    features, errors, groups, _ = sample_predictions()
    # This adapter never exposes a model-native score. Its predictions only use local gallery density.
    common = features[["gallery_count"]].astype(float)
    common.loc[::8, "gallery_count"] = np.nan
    model_path = tmp_path / "confidence.joblib"
    model = GeoConfidence(["gallery_count"]).fit(common, errors, groups=groups)
    queries = pd.DataFrame({"gallery_count": [1.0, np.nan, 42.0]}, index=[99, 51, 2])
    before = model.predict_proba(queries)
    assert before.shape == (3,)
    assert np.all(np.isfinite(before) & (before >= 0.0) & (before <= 1.0))
    with_truth = queries.assign(error_km=[0.0, 300.0, 0.0], true_latitude=[47.0, 50.0, 60.0])
    np.testing.assert_array_equal(model.predict_proba(with_truth), before)
    model.save(model_path)
    restored = GeoConfidence.load(model_path)
    np.testing.assert_array_equal(restored.predict_proba(queries), before)
    assert set(restored.fit_groups_) == set(model.fit_groups_)
    assert set(restored.calibration_groups_) == set(model.calibration_groups_)


def test_per_model_parameters_learn_opposite_meanings_of_a_score():
    features, errors, groups, _ = sample_predictions()
    first = GeoConfidence(["native_score"]).fit(features, errors, groups=groups)
    second = GeoConfidence(["native_score"]).fit(features, np.where(errors == 4.0, 60.0, 4.0), groups=groups)
    queries = pd.DataFrame({"native_score": [-0.9, 0.9]})
    first_prob = first.predict_proba(queries)
    second_prob = second.predict_proba(queries)
    assert first_prob[0] < first_prob[1]
    assert second_prob[0] > second_prob[1]
    assert set(first.calibration_groups_) == set(second.calibration_groups_)


def test_single_class_calibration_uses_only_its_own_labels():
    features, errors, groups, _ = sample_predictions()
    model = GeoConfidence(["gallery_count"]).fit(features, errors, groups=groups)
    held_out = np.isin(groups, model.calibration_groups_)
    # Every calibration row hits, while every row in base-model fitting misses.
    model.fit(features, np.where(held_out, 1.0, 60.0), groups=groups)
    expected = (held_out.sum() + 1) / (held_out.sum() + 2)
    np.testing.assert_allclose(model.predict_proba(features.iloc[:4]), expected)


def test_prediction_rejects_missing_or_invalid_declared_features():
    features, errors, groups, _ = sample_predictions()
    model = GeoConfidence(["native_score"]).fit(features, errors, groups=groups)
    invalid_inputs = [
        pd.DataFrame({"other": [3]}),
        pd.DataFrame({"native_score": [np.inf]}),
        pd.DataFrame({"native_score": ["unknown"]}),
    ]
    for invalid in invalid_inputs:
        with pytest.raises(ValueError):
            model.predict_proba(invalid)


def test_gallery_features_count_photos_across_the_dateline_and_preserve_query_order():
    # Two photos share a location 2.22 km across the dateline from the first query.
    gallery = [[0.0, 179.99], [0.0, 179.99], [0.0, 179.77]]
    queries = [[0.0, -179.99], [0.0, 179.77], [0.0, -179.99]]
    features = GalleryFeatureExtractor(gallery).transform(queries, [True, False, False])
    np.testing.assert_allclose(features.nearest_train_km, [2.224, 0.0, 2.224], atol=0.001)
    np.testing.assert_allclose(features.log1p_train_5km, np.log1p([2, 1, 2]))
    np.testing.assert_allclose(features.log1p_train_25km, np.log1p([2, 3, 2]))
    np.testing.assert_array_equal(features.has_date, [True, False, False])
