import json

import joblib
import numpy as np
import pandas as pd
import pytest

from lib.confidence import GeoConfidence
from models.t6_router.geographic_gate import GeographicGate, NONPOSITION, POSITION
from models.t6_router import region_gate
from scripts.confidence.evaluate import FEATURES


def rows(label, start):
    table = pd.DataFrame({"id": np.arange(start, start + 6), "region": label,
                          "block": [f"{label}1"] * 3 + [f"{label}2"] * 3,
                          "T4_lat_pred": 46., "T4_lon_pred": 7., "T5_lat_pred": 47.,
                          "T5_lon_pred": 8., "T4_error_km": [1, 40, 1, 40, 1, 40],
                          "T5_error_km": [40, 1, 1, 40, 40, 1]})
    for c in FEATURES["T4"] + FEATURES["T5"]:
        table[c] = 1.
    return table


def test_four_directional_counts_and_net_hits():
    frame = rows("final", 40)
    # T4-only rows 0,4 flip opposite ways; T5-only rows 1,5 both flip to T4.
    old = np.array([False, False, True, True, True, False])
    new = ~old
    result = region_gate.paired_change(frame, new, old)
    assert result["additional_T4_only_rescues"] == 1
    assert result["previous_T4_only_rescues_reversed"] == 1
    assert result["additional_T5_only_losses"] == 2
    assert result["previous_T5_only_losses_recovered"] == 0
    assert result["net_T4_only_rescues"] == 0
    assert result["net_T5_only_losses"] == 2
    assert result["net_hits_25km"] == -2


def test_entire_regions_held_out_and_final_labels_only_used_for_scoring(tmp_path, monkeypatch):
    dev = pd.concat([rows(name, i * 10 + 1) for i, name in enumerate(("A", "B", "C"))])
    final = rows("D", 100)
    assets = {"gate": tmp_path / "old.joblib"}
    joblib.dump(GeographicGate(POSITION + NONPOSITION, None, .51), assets["gate"])
    for model in ("T4", "T5"):
        key = model.lower() + "_confidence"
        assets[key] = tmp_path / (key + ".joblib")
        GeoConfidence(FEATURES[model]).fit(dev, dev[f"{model}_error_km"], groups=dev.block).save(assets[key])
    seen = []
    original = region_gate.fit_confidence

    def capture(frame):
        seen.append(frozenset(frame.region))
        return original(frame)

    monkeypatch.setattr(region_gate, "fit_confidence", capture)
    outputs = []
    for i, held in enumerate((final, final.assign(T4_error_km=0., T5_error_km=100.))):
        folder = tmp_path / f"run{i}"
        report = region_gate.run_region_gate(dev, held, bundle_assets=assets, output_dir=folder)
        assert seen[-4:] == [frozenset("BC"), frozenset("AC"), frozenset("AB"), frozenset("ABC")]
        assert report["n"] == len(final)
        assert report == json.loads((folder / "summary.json").read_text())
        assert (folder / "regional_gate.joblib").is_file()
        chosen = pd.read_parquet(folder / "decisions.parquet")
        assert np.array_equal(chosen.id, final.id)
        outputs.append(chosen.filter(regex="p25$|p_T4$|choose_T4$"))
    pd.testing.assert_frame_equal(*outputs)


def test_guard_against_final_overlap(tmp_path):
    dev = pd.concat([rows("A", 1), rows("B", 10)])
    for final in (rows("C", 1), rows("B", 99)):
        with pytest.raises(ValueError, match="overlap"):
            region_gate.run_region_gate(dev, final, bundle_assets={}, output_dir=tmp_path / "run")


def test_confidence_with_single_training_block():
    only = rows("A", 1).assign(block="1", T5_error_km=100.)
    models = region_gate.fit_confidence(only)
    assert len(models) == 2
    np.testing.assert_allclose(region_gate.confidence_scores(rows("B", 10), models)[:, 1], 1 / 8)
