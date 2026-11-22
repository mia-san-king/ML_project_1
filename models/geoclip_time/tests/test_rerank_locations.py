"""Exercise T1 location reranking with real small towers and shuffled caches."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


def runner():
    name = "models.geoclip_time.scripts.rerank_locations"
    assert importlib.util.find_spec(name) is not None, "location reranking CLI is not implemented"
    return importlib.import_module(name)


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    from models.geoclip_time.lib.temporal_data import load_temporal_data
    from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP
    from models.geoclip_time.lib import temporal_training

    root = tmp_path / "data"
    (root / "features").mkdir(parents=True)
    (root / "meta").mkdir()
    ids = np.array([15, 10, 12, 16, 17, 13, 11, 14])
    splits = {10: "train", 11: "train", 17: "train", 18: "train", 12: "val",
              13: "test_in", 14: "test_city", 15: "test_region", 16: "nadir"}
    dates = {10: "1900-01-01", 11: "2000-01-01", 17: "1950-01-01", 18: "1900-01-01",
             12: "2000-06-01", 13: None, 14: "1960-01-01", 15: "2030-01-01", 16: "2000-01-01"}
    metadata_ids = list(reversed([*ids, 18]))
    metadata = pd.DataFrame({"id": metadata_ids, "latitude": [0. if i == 10 else 10. for i in metadata_ids],
                            "longitude": 0., "date_shot_min": [dates[i] for i in metadata_ids],
                            "date_shot_max": [dates[i] for i in metadata_ids]})
    metadata.to_parquet(root / "meta/smapshot.parquet", index=False)
    split_path = tmp_path / "splits.csv"
    pd.DataFrame({"id": metadata_ids, "split": [splits[i] for i in metadata_ids]}).to_csv(split_path, index=False)
    features = np.zeros((len(ids), 768), dtype=np.float32)
    features[:, :2] = [1., .99]
    np.save(root / "features/clip_vitl14.npy", features)
    np.save(root / "features/clip_vitl14_ids.npy", ids)
    time_config = {"year_center": 1950., "year_scale": 100., "min_year": 1800, "max_year": 2026}
    data = load_temporal_data(root, split_path, time_config=time_config)

    def factory(architecture, *, pretrained):
        model = TemporalGeoCLIP(nn.Linear(768, 512), nn.Linear(2, 512), mode=architecture["mode"], hidden_dim=4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.image_mlp.weight[0, 0] = model.image_mlp.weight[1, 1] = 1
            model.location_encoder.bias[0] = 1
            model.location_encoder.weight[0, 0] = -.1
            model.location_encoder.weight[1, 0] = .1
        return model

    architecture = {"mode": "both", "feature_dim": 768, "embed_dim": 512, "time_dim": 3, "hidden_dim": 4}
    payload = {"format_version": 1, "architecture": architecture, "time_config": time_config,
               "model_state": factory(architecture, pretrained=False).state_dict(),
               "config": {"features": "clip_vitl14", "allow_partial_features": True, "model": architecture,
                          "time": time_config, "run_name": "tiny_trained_T1"},
               "validation_ids": [12], "epoch": 1, "history": [{"epoch": 1}], "audit": data.audit}
    checkpoint = tmp_path / "t1.pt"
    torch.save(payload, checkpoint)
    actual_load = temporal_training.load_temporal_checkpoint
    monkeypatch.setattr(temporal_training, "load_temporal_checkpoint",
                        lambda path, *, device: actual_load(path, device=device, model_factory=factory))
    return root, split_path, checkpoint, data, payload


def test_runner_calls_all_queries_once_and_keeps_photo_ids_at_duplicate_gps(local_inputs, tmp_path, monkeypatch):
    cli = runner()
    from models.geoclip_time.lib import temporal_location_retrieval

    root, split, checkpoint, data, _ = local_inputs
    calls = []
    real_topk = temporal_location_retrieval.query_time_location_topk

    def record_real_topk(model, features, times, gps, **kwargs):
        assert not model.training and all(not parameter.requires_grad for parameter in model.parameters())
        assert model.logit_scale.device.type == "cpu"
        calls.append((np.asarray(times).copy(), np.asarray(gps).copy()))
        return real_topk(model, features, times, gps, **kwargs)

    monkeypatch.setattr(temporal_location_retrieval, "query_time_location_topk", record_real_topk)
    output = tmp_path / "output"
    result = cli.run_retrieval(data_root=root, split_csv=split, checkpoint=checkpoint,
                               output_dir=output, threads=1, query_chunk=2, reference_chunk=2)
    assert len(calls) == 1
    query_mask = data.table.split.isin(["val", "test_in", "test_city", "test_region"])
    np.testing.assert_array_equal(calls[0][0], data.times[query_mask])
    assert calls[0][1].shape == (3, 2)  # Two different photos share [10, 0].
    assert len(np.unique(calls[0][1], axis=0)) == 2
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert set(predictions.method) == {"location", "year"}
    assert "spatial_cosine" in predictions and "visual_score" not in predictions
    assert len(predictions) == 8 and set(predictions.id) == {12, 13, 14, 15}
    assert result["selected_gamma"] == .05
    assert predictions.query("method == 'location'").neighbor_id.tolist() == [10] * 4
    year_ids = predictions.query("method == 'year'").set_index("id").neighbor_id.to_dict()
    assert year_ids == {12: 11, 13: 10, 14: 17, 15: 11}
    np.testing.assert_allclose(predictions.score, predictions.spatial_cosine - predictions.gamma * predictions.year_penalty)
    assert predictions.query("id == 15").query_year.eq(2030.).all()  # Archive year known; T1 time is zero.
    assert predictions.query("id == 13").year_penalty.eq(0).all()
    assert result["references"]["n"] == 3
    assert result["candidate_spread"]["val"]["candidate_radius_median_km"] > 1000
    assert json.loads((output / "summary.json").read_text())["selected_gamma"] == .05
    assert json.loads((output / "val_gamma_search.json").read_text())["selection_rows"] == 1
    with pytest.raises(FileExistsError):
        cli.run_retrieval(data_root=root, split_csv=split, checkpoint=checkpoint, output_dir=output)


@pytest.mark.parametrize("fault, match", [("mode", "mode=both"), ("time", "time"), ("coverage", "audit"), ("partial", "partial")])
def test_runner_rejects_checkpoint_and_input_mismatches(local_inputs, tmp_path, fault, match):
    cli = runner()
    root, split, checkpoint, _, original = local_inputs
    payload = deepcopy(original)
    if fault == "mode":
        # A coherent T0 checkpoint must be rejected before inference.
        payload["architecture"]["mode"] = "none"
        payload["config"]["model"]["mode"] = "none"
        payload["model_state"] = {key: value for key, value in payload["model_state"].items() if "time_adapter" not in key}
    elif fault == "time":
        payload["time_config"] = {**payload["time_config"], "year_center": 1900.}
    elif fault == "coverage":
        payload["audit"]["coverage"]["train"]["expected"] += 1
    else:
        payload["config"]["allow_partial_features"] = False
    torch.save(payload, checkpoint)
    output = tmp_path / fault
    with pytest.raises(ValueError, match=match):
        cli.run_retrieval(data_root=root, split_csv=split, checkpoint=checkpoint, output_dir=output, threads=1)
    assert not output.exists()


def test_candidate_spread_uses_the_first_candidate_as_anchor_and_all_candidates():
    cli = runner()
    gps = np.array([[0., 0.], [0., 0.], [0., 1.], [0., 2.]])
    actual = cli.candidate_spread_summary(np.array([[2, 0, 3], [0, 1, 0]]), gps, chunk=1)
    assert actual["n"] == 2
    assert actual["candidate_radius_median_km"] == pytest.approx(55.5975401168)
    assert actual["candidate_radius_le5km_percent"] == 50.


def test_help_runs_without_scientific_packages():
    cli = runner()
    completed = subprocess.run([sys.executable, "-S", str(Path(cli.__file__)), "--help"], capture_output=True, text=True)
    assert completed.returncode == 0
    assert "--reference-chunk" in completed.stdout and "--device" in completed.stdout
