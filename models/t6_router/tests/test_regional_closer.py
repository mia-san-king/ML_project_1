"""The routing target is great-circle error ordering, independent of the 25 km cutoff."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from lib.confidence import GeoConfidence
from models.t6_router import regional_closer
from models.t6_router.geographic_gate import GeographicGate, NONPOSITION, POSITION
from models.t6_router.region_gate import run_region_gate
from scripts.confidence.evaluate import FEATURES


def region_frame(region, first_id):
    count = 8
    frame = pd.DataFrame({
        "id": np.arange(first_id, first_id + count), "region": region,
        "block": [region + "0"] * 4 + [region + "1"] * 4,
        "T4_lat_pred": 46., "T4_lon_pred": 7., "T5_lat_pred": 47., "T5_lon_pred": 8.,
        "T4_error_km": [35., 15., 5., 60., 70., 45., 12., 25.],
        "T5_error_km": [80., 3., 41., 6., 45., 75., 12., 25.],
        "T4_p25": .3, "T5_p25": .6,
    })
    for col in FEATURES["T4"] + FEATURES["T5"]:
        frame[col] = 1.
    return frame


@pytest.mark.parametrize("e4,e5,expected", [(35., 80., 1.), (15., 3., 0.), (5., 41., 1.),
                                             (60., 6., 0.), (25., 25., .5)])
def test_training_target_uses_each_strict_distance_order(e4, e5, expected):
    # A single observed class determines the fallback exactly. Ties provide no labels.
    sample = region_frame("A", 1).iloc[:1].copy()
    sample["T4_error_km"], sample["T5_error_km"] = e4, e5
    model = regional_closer.fit_closer(sample)
    np.testing.assert_array_equal(model.predict_t4_probability(sample[list(POSITION + NONPOSITION)]), [expected])


def test_paired_counts_can_improve_accuracy_while_mean_distance_worsens():
    sample = region_frame("D", 100)
    old = np.array([False, True, True, False, True, False, False, True])
    new = np.array([True, False, False, True, True, True, True, False])
    change = regional_closer.paired_distances(sample, new, old)
    expected = {"switches": 7, "T5_to_T4": 4, "T4_to_T5": 3, "improved": 3, "worsened": 2,
                "equal_after_switch": 2, "saved_km": 87, "added_km": 90, "net_saved_km": -3,
                "mean_saved_km": -3 / 8, "net_correct_non_ties": 1,
                "T4_closer_newly_correct": 2, "T4_closer_previously_correct_lost": 1,
                "T5_closer_newly_correct": 1, "T5_closer_previously_correct_lost": 1}
    for column, value in expected.items():
        assert change[column] == pytest.approx(value)
    metrics = regional_closer.distance_metrics(sample, new)
    for key, value in {"n": 8, "n_non_ties": 6, "n_ties": 2, "correct_non_ties": 3,
                       "accuracy_non_ties_pct": 50, "choose_T4": 5, "mean_km": 291 / 8, "median_km": 38,
                       "mean_oracle_regret_km": 115 / 8, "hits_25km": 3, "hit_25km_pct": 37.5}.items():
        assert metrics[key] == pytest.approx(value)
    tied = sample.assign(T4_error_km=0., T5_error_km=0.)
    assert regional_closer.distance_metrics(tied, old)["accuracy_non_ties_pct"] is None


def test_unseen_labels_do_not_change_predictions_and_evidence_matches_ids(tmp_path):
    development = pd.concat([region_frame(name, 10 * i + 1) for i, name in enumerate("ABC")],
                            ignore_index=True)
    final = region_frame("D", 100).sample(frac=1., random_state=5).reset_index(drop=True)
    assets = {"gate": tmp_path / "current.joblib"}
    joblib.dump(GeographicGate(POSITION + NONPOSITION, None, .51), assets["gate"])
    for model in ("T4", "T5"):
        path = tmp_path / f"{model}.joblib"
        GeoConfidence(FEATURES[model]).fit(development, development[f"{model}_error_km"],
                                           groups=development.block).save(path)
        assets[f"{model.lower()}_confidence"] = path
    predicted, scored = [], []
    for i, held in enumerate((final, final.assign(T4_error_km=0., T5_error_km=100.))):
        ref, out = tmp_path / f"reference{i}", tmp_path / f"closer{i}"
        run_region_gate(development, held, bundle_assets=assets, output_dir=ref)
        report = regional_closer.run_closer(development, held, reference_dir=ref, output_dir=out)
        assert report == json.loads((out / "summary.json").read_text())
        assert (out / "closer_gate.joblib").is_file()
        decisions = pd.read_parquet(out / "decisions.parquet").set_index("id").sort_index()
        baseline = pd.read_parquet(ref / "decisions.parquet").set_index("id").sort_index()
        truths = held.set_index("id").sort_index()
        assert decisions.index.equals(truths.index)
        for expert, other in (("T4", "T5"), ("T5", "T4")):
            errors = truths[f"{expert}_error_km"]
            fixed = report["methods"]["always_" + expert]
            assert fixed["choose_T4"] == (len(truths) if expert == "T4" else 0)
            assert fixed["mean_km"] == pytest.approx(errors.mean())
            assert fixed["median_km"] == pytest.approx(errors.median())
            assert fixed["hits_25km"] == (errors <= 25).sum()
            assert fixed["correct_non_ties"] == (errors < truths[f"{other}_error_km"]).sum()
        policies = ("current_T6", "regional_T6", "regional_closer")
        for model in policies:
            name = model + "_choose_T4"
            if model != "regional_closer":
                np.testing.assert_array_equal(decisions[name], baseline[name])
            km = np.where(decisions[name], truths.T4_error_km, truths.T5_error_km)
            assert report["methods"][model]["mean_km"] == pytest.approx(np.mean(km))
        for model in policies[:2]:
            delta = report["comparisons"]["vs_" + model]
            mean_saved = report["methods"][model]["mean_km"] - report["methods"]["regional_closer"]["mean_km"]
            assert delta["mean_saved_km"] == pytest.approx(mean_saved)
            gained4 = decisions.regional_closer_choose_T4 & ~decisions[model + "_choose_T4"]
            gained5 = ~decisions.regional_closer_choose_T4 & decisions[model + "_choose_T4"]
            only4 = truths.T4_error_km.le(25) & truths.T5_error_km.gt(25)
            only5 = truths.T5_error_km.le(25) & truths.T4_error_km.gt(25)
            hit_changes = delta["hit25_changes"]
            assert hit_changes["additional_T4_only_rescues"] == (only4 & gained4).sum()
            assert hit_changes["previous_T4_only_rescues_reversed"] == (only4 & gained5).sum()
            assert hit_changes["additional_T5_only_losses"] == (only5 & gained4).sum()
            assert hit_changes["previous_T5_only_losses_recovered"] == (only5 & gained5).sum()
            assert hit_changes["net_hits_25km"] == (report["methods"]["regional_closer"]["hits_25km"]
                                                    - report["methods"][model]["hits_25km"])
        predicted.append(decisions.filter(regex="p_T4$|choose_T4$|p25$"))
        scored.append(report["methods"]["regional_closer"]["hits_25km"])
    pd.testing.assert_frame_equal(*predicted)
    assert scored[0] != scored[1]
