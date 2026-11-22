"""Tests for the validation-fitted weight and query-time T1 candidate scores."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP
from models.t5_reranker.rerank import choose_candidates, location_cosines, select_weight


def test_zero_weight_preserves_full_precision_t3_and_earliest_visual_rank():
    base = np.array([[0.75, 0.75 + 1e-12, 0.4], [0.25, 0.75, 0.75]])
    location = np.array([[1, -1, 0], [1, -1, 1]], dtype=np.float32)
    np.testing.assert_array_equal(choose_candidates(base, location, 0), [1, 1])


def test_weighted_sum_selects_winner_within_each_candidate_list():
    base = np.array([[0.75, 0.50], [0.25, 0.75], [0.50, 0.25]])
    location = np.array([[0, 0.75], [0.50, -0.50], [0.75, -0.25]])
    # The second row ties at 0.50; the first visual candidate wins.
    np.testing.assert_array_equal(choose_candidates(base, location, 0.5), [1, 0, 0])


def validation_inputs():
    truth = np.tile([46.5, 7.0], (3, 1))
    gps = np.repeat(truth[:, None, :], 2, axis=1)
    gps[:, :, 1] += [[0.15, 0.40], [0.70, 0.45], [0.90, 1.50]]
    base = np.tile([0.8, 0.5], (3, 1))
    location = np.tile([0.0, 0.6], (3, 1))
    return base, location, gps, truth


def test_weight_optimizes_full_validation_median_and_breaks_ties_by_weight():
    # The second candidate has a lower median error, worse mean and fewer hits.
    # Both nonzero weights choose it for all rows. The smaller one wins.
    weight, curve = select_weight(*validation_inputs(), np.array(["val"] * 3), [1.0, 0.0, 0.75])
    assert weight == 0.75
    assert len(curve) == 3


@pytest.mark.parametrize("splits", [["test_in"] * 3, ["val", "test_city", "val"]])
def test_weight_selection_rejects_test_labels(splits):
    with pytest.raises(ValueError):
        select_weight(*validation_inputs(), np.array(splits), [0.0, 0.75])


class AddDate(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, base, times):
        return base + self.factor * times[..., :2]


def test_per_query_cosines_match_independent_t1_scores_without_logit_scaling():
    model = TemporalGeoCLIP(nn.Identity(), nn.Identity(), embed_dim=2, mode="both").eval()
    model.image_time_adapter = AddDate(0.5)
    model.location_time_adapter = AddDate(-1.0)
    features = torch.tensor([[0.3, 1.0], [0.3, 1.0], [1.0, 0.1]])
    dates = torch.tensor([[0.5, 0.1, 1.0], [-0.9, 0.7, 1.0], [0.0, 0.0, 0.0]])
    gps = torch.tensor([[[1.0, 0.2], [0.2, 0.6]], [[1.0, 0.2], [0.2, 0.6]],
                        [[-0.1, -0.4], [1.0, 0.2]]])
    with torch.no_grad():
        bases = model.encode_location_base(gps.reshape(-1, 2)).reshape_as(gps)
        actual = location_cosines(model, features, dates, bases)
        expected = torch.cat([model.score_candidates(features[i:i + 1], dates[i:i + 1], gps[i])
                              for i in range(len(features))]) / model.logit_scale.exp()
    assert not torch.allclose(expected[0], expected[1])
    torch.testing.assert_close(actual, expected)


def test_new_photo_prediction_uses_train_parents_and_accepts_missing_dates(tmp_path, monkeypatch):
    from models.geoclip_time.scripts.rerank_references import _ids_hash, _sha256
    from models.t5_reranker import predict_photo as inference

    # Only parent 1's synthetic variant exactly matches the new query.
    # A held-out photo also matches; it must never enter the gallery.
    ids = np.array([999, *range(1, 51), -1], dtype=np.int64)
    features = np.zeros((len(ids), 768), dtype=np.float32)
    features[:, :2] = [0.8, 1.0]
    features[ids == 1, :2] = [0.0, 1.0]
    features[np.isin(ids, [-1, 999]), :2] = [1.0, 0.0]
    latitude = np.where(np.abs(ids) == 1, 46.6, 47.0)
    date = np.where(np.abs(ids) == 1, "1900-01-01", "2000-01-01")
    meta = pd.DataFrame({"id": ids, "latitude": latitude, "longitude": 7.0,
                         "date_shot_min": date, "date_shot_max": date})
    (tmp_path / "features").mkdir()
    (tmp_path / "meta").mkdir()
    files = {"metadata": tmp_path / "meta/smapshot.parquet", "features": tmp_path / "features/clip_vitl14.npy",
             "feature_ids": tmp_path / "features/clip_vitl14_ids.npy", "split": tmp_path / "splits.csv"}
    meta.to_parquet(files["metadata"], index=False)
    np.save(files["features"], features)
    np.save(files["feature_ids"], ids)
    pd.DataFrame({"id": ids, "split": np.where(ids == 999, "val", "train")}).to_csv(files["split"], index=False)

    projection = nn.Linear(768, 2, bias=False)
    with torch.no_grad():
        projection.weight.zero_()
        projection.weight[:, :2] = torch.eye(2)
    towers = {mode: TemporalGeoCLIP(projection, nn.Identity(), embed_dim=2, mode=mode).eval()
              for mode in ("none", "both")}
    monkeypatch.setattr(inference, "restore_tower", lambda section, device: towers[section["mode"]].to(device))
    package = {"model": "T5", "format_version": 1, "t0": {"mode": "none", "time_config": {}},
               "t1": {"mode": "both", "time_config": {}}, "weight": 0.0, "gamma": 1.0, "top_k": 50,
               "parent_grouping": True, "gallery_ids_sha256": _ids_hash(np.arange(1, 51)),
               "input_sha256": {name: _sha256(path) for name, path in files.items()}}
    model_path = tmp_path / "T5.pt"
    torch.save(package, model_path)
    query_path = tmp_path / "query.npy"
    query = features[-1].copy()
    inputs = dict(model_path=model_path, data_root=tmp_path, split_csv=files["split"], feature=query_path)
    np.save(query_path, query)
    undated = inference.predict_photo(**inputs)
    assert (undated["neighbor_id"], undated["lat_pred"], undated["lon_pred"]) == (1, 46.6, 7.0)

    np.save(query_path, query[None, :])
    dated = inference.predict_photo(**inputs, date_min="2000-01-01", date_max="2000-01-01")
    assert dated["neighbor_id"] == 2
    assert dated["visual_rank"] == 2
    assert dated["score"] == dated["t3_score"]

    gallery = inference.prepare_gallery(package, tmp_path, files["split"])
    np.testing.assert_array_equal(gallery["ids"], np.arange(1, 51))
    assert gallery["vectors"].shape == (51, 2)  # 50 real parents + 1 training variant
    assert gallery["groups"][-1] == 0
    gallery_path = tmp_path / "gallery.npz"
    np.savez(gallery_path, **gallery)
    indexed = dict(model_path=model_path, gallery_path=gallery_path, feature=query_path)
    files["metadata"].rename(tmp_path / "archived_metadata.parquet")
    assert inference.predict_indexed_photo(**indexed) == undated
    assert inference.predict_indexed_photo(**indexed, date_min="2000-01-01", date_max="2000-01-01") == dated
