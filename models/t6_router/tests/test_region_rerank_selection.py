import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from models.geoclip_time.scripts.rerank_references import DEFAULT_GAMMAS, _ids_hash
from models.t5_reranker.rerank import WEIGHTS
from models.t6_router import region_experts as expert


def test_allowed_val_chooses_both_rerank_parameters_before_reading_query_truth(tmp_path):
    # All visual scores prefer the older, western parent. The val labels are
    # all eastern, and the query truth is deliberately changed between calls.
    ids = [10, 20, 300, 301, 400, 500]
    roles = ["train", "train", "val", "val", "query", "ignore"]
    dates = ["1900-01-01", "1975-01-01", *(["1975-01-01"] * 4)]
    rows = pd.DataFrame({"id": ids, "feat_idx": range(6), "split": roles,
                         "latitude": [46.] * 6, "longitude": [6., 8., 8., 8., 6., 6.],
                         "date_shot_min": dates, "date_shot_max": dates})
    raw = np.array([[1., 0.], [.9, (1 - .9**2)**.5], *([[1., 0.]] * 4)], dtype=np.float32)
    data = SimpleNamespace(table=rows, times=np.zeros((6, 3), np.float32), features=raw)

    class T0:
        embed_dim, time_dim = 2, 3

        def encode_image(self, features, times):
            return features

    class T1(T0):
        mode = "both"

        def encode_location_base(self, gps):
            return torch.ones_like(gps)

        def location_time_adapter(self, bases, times):
            return bases

    path = tmp_path / "fit.json"
    def predict():
        return expert._predict_t5(data, *expert._partition(rows), T0(), T1(),
                                  device="cpu", top_k=2, selection_path=path)

    first = predict()
    evidence = path.read_bytes()
    audit = json.loads(evidence)
    assert audit["ids"] == [300, 301] and audit["n"] == 2
    assert audit["ids_sha256"] == _ids_hash([300, 301])
    assert [p["gamma"] for p in audit["gamma_curve"]] == list(DEFAULT_GAMMAS)
    assert [p["weight"] for p in audit["weight_curve"]] == list(WEIGHTS)
    medians = [r["median_km"] for r in audit["gamma_curve"]]
    assert audit["gamma"] == DEFAULT_GAMMAS[int(np.argmin(medians))] > 0.
    assert audit["weight"] == 0.  # All location scores tie.
    rows.loc[rows.split.isin(["query", "ignore"]), "longitude"] = 8.
    second = predict()
    assert path.read_bytes() == evidence
    assert first.T5_neighbor_id.tolist() == second.T5_neighbor_id.tolist() == [20]
    assert first.T5_error_km.iloc[0] > 100 and second.T5_error_km.iloc[0] == 0.
    rows.loc[rows.split.eq("val"), "split"] = "ignore"
    with pytest.raises(ValueError, match="val"):
        predict()


@pytest.mark.parametrize("cached_refit", [False, True])
def test_cached_predictions_reject_other_selection_protocol(tmp_path, monkeypatch, cached_refit):
    monkeypatch.setattr(expert, "fit_towers", lambda *a, **kw: {})
    border = tmp_path / "data/local_runs/t6_aug8192/bundle_v2/ne_admin0.gpkg"
    border.parent.mkdir(parents=True)
    border.write_bytes(b"border")
    split = tmp_path / "roles.csv"
    split.write_bytes(b"split")
    artifacts = tmp_path / "run/experts"
    artifacts.mkdir(parents=True)
    (artifacts / "predictions.parquet").write_bytes(b"predictions")
    report = artifacts / "rerank_validation.json"
    report.write_bytes(b"evidence")
    inputs = {"split_sha256": expert._sha256(split), "border_sha256": expert._sha256(border)}
    if cached_refit:
        inputs["rerank_selection"] = expert.RERANK_SELECTION
    receipt = {"inputs": inputs, "predictions_sha256": expert._sha256(artifacts / "predictions.parquet"),
               "rerank_validation_sha256": expert._sha256(report)}
    (artifacts / "summary.json").write_text(json.dumps(receipt))
    def run(refit):
        return expert.run_experts(tmp_path, tmp_path, split, artifacts.parent, device="cpu", refit_rerank=refit)
    assert run(cached_refit) == artifacts / "predictions.parquet"
    with pytest.raises(ValueError, match="protocol"):
        run(not cached_refit)
    if cached_refit:
        report.write_bytes(b"different evidence")
        with pytest.raises(ValueError, match="protocol"):
            run(True)
