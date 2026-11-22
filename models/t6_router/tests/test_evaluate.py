import joblib
import numpy as np
import pandas as pd
import pytest

from models.t6_router import evaluate
from models.t6_router.geographic_gate import GeographicGate, NONPOSITION, POSITION


def test_batch_uses_same_learned_rule_as_photo_and_keeps_archived_evaluation(tmp_path, monkeypatch):
    # Every T4 confidence is lower. The learned gate favors T4 anyway.
    truth = pd.DataFrame({"split": np.repeat(evaluate.TEST_SPLITS, 2), "block": np.arange(6),
                          "latitude": 46., "longitude": 7.}, index=pd.Index(range(1, 7), name="id"))
    monkeypatch.setattr(evaluate, "original_rows", lambda *args: (truth, None, {}))
    rows = []
    for name, lon, confidence in (("T4", 7., .1), ("T5", 8., .9)):
        frame = truth[["split", "block"]].assign(model=name, lat_pred=46., lon_pred=lon,
                                                 p_within_25km=confidence, confidence_partition="test")
        frame["error_km"] = evaluate._errors(truth[list(evaluate.GPS)].to_numpy(),
                                            frame[list(evaluate.PREDICTED)].to_numpy())
        rows.append(frame.reset_index())
    path = tmp_path / "predictions.parquet"
    pd.concat(rows).to_parquet(path, index=False)
    gate_path = tmp_path / "gate.joblib"
    joblib.dump(GeographicGate(POSITION + NONPOSITION, None, .8), gate_path)
    monkeypatch.setattr(evaluate, "load_bundle", lambda model_path: {"gate": gate_path})
    output = tmp_path / "v2"
    report = evaluate.run(path, None, None, output)
    results = pd.read_parquet(output / "predictions.parquet")
    assert results.chosen_model.eq("T4").all() and results.error_km.eq(0).all()
    assert results.gate_p_T4_given_one_hit.eq(.8).all()
    assert report["rule"] == evaluate.RULE and "gate" in report["inputs"]
    assert all(row["rescued_hits_25km"] == 2 for row in report["routing"])
    with pytest.raises(FileExistsError):
        evaluate.run(path, None, None, output, legacy_confidence=True)
    legacy_dir = tmp_path / "legacy"
    old_report = evaluate.run(path, None, None, legacy_dir, legacy_confidence=True)
    assert old_report["rule"] == evaluate.LEGACY_RULE
    assert pd.read_parquet(legacy_dir / "predictions.parquet").chosen_model.eq("T5").all()
