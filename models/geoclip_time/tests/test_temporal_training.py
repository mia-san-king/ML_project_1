"""Exercise the complete runner with small real torch towers and local caches."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


class CountingLocation(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 4))
        self.eval_rows = 0

    def forward(self, gps):
        if not self.training:
            self.eval_rows += len(gps)
        return super().forward(gps)


def tiny_factory(architecture, *, pretrained):
    from models.geoclip_time.lib.temporal_model import TemporalGeoCLIP

    image = nn.Linear(768, 4)
    location = CountingLocation()
    location[0].bias.requires_grad_(False)
    return TemporalGeoCLIP(
        image, location, embed_dim=4, hidden_dim=8,
        mode=architecture["mode"], time_dim=3,
    )


@pytest.fixture
def fixture_data(tmp_path):
    root = tmp_path / "data"
    for name in ("features", "meta", "geo"):
        (root / name).mkdir(parents=True)
    # Cache order deliberately differs from metadata and split order.
    ids = np.array([40, 12, 31, 11, 30, 13, 50])
    np.save(root / "features/clip_vitl14_ids.npy", ids)
    features = np.zeros((7, 768), dtype=np.float32)
    features[:, :4] = np.random.default_rng(3).normal(size=(7, 4))
    np.save(root / "features/clip_vitl14.npy", features)
    meta = pd.DataFrame({
        "id": [11, 12, 13, 30, 31, 40, 50],
        "latitude": [46.1, 46.2, 46.3, 46.4, 46.5, 46.6, 46.7],
        "longitude": [6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7],
        "date_shot_min": ["1900-01-01", None, "1950-01-01", "1910-01-01", "1980-01-01", "2000-01-01", "2010-01-01"],
        "date_shot_max": ["1902-01-01", None, "1950-01-01", "1912-01-01", "1980-01-01", "2000-01-01", "2010-01-01"],
        "view_type": ["oblique"] * 7,
    })
    meta.to_parquet(root / "meta/smapshot.parquet", index=False)
    splits = meta[["id"]].copy()
    splits["split"] = ["train", "train", "train", "val", "val", "test_in", "test_city"]
    splits["block"] = np.arange(7)
    split_path = tmp_path / "split.csv"
    splits.to_csv(split_path, index=False)
    np.save(root / "geo/grid_1km.npy", meta[["latitude", "longitude"]].to_numpy(np.float32))
    cfg = {
        "run_name": "tiny", "data_root": str(root), "split_csv": str(split_path),
        "epochs": 1, "batch_size": 2, "n_neg": 2, "val_limit": 2,
        "seed": 3, "query_chunk": 1, "gallery_chunk": 3,
        "model": {"backbone": "tiny-test", "embed_dim": 4, "hidden_dim": 8, "mode": "both"},
        "time": {"year_center": 1950.0, "year_scale": 100.0, "min_year": 1800, "max_year": 2026},
    }
    return root, split_path, cfg


def runner():
    return importlib.import_module("models.geoclip_time.lib.temporal_training")


def test_real_training_checkpoint_reloads_identical_predictions_and_processes_tail(fixture_data):
    root, split_path, cfg = fixture_data
    torch.manual_seed(cfg["seed"])
    initial = tiny_factory(cfg["model"], pretrained=True)
    result = runner().run_training(cfg, model_factory=tiny_factory)
    assert not torch.equal(initial.image_mlp.weight, result["model"].image_mlp.weight)
    assert not torch.equal(initial.location_encoder[2].weight, result["model"].location_encoder[2].weight)
    assert torch.equal(initial.location_encoder[0].bias, result["model"].location_encoder[0].bias)
    log = json.loads(result["log"].read_text())
    assert log["history"][0]["train_examples"] == 3
    assert log["history"][0]["optimizer_steps"] == 2
    assert log["history"][0]["last_batch_size"] == 1
    assert np.isfinite(log["history"][0]["train_loss"])
    assert set(log["validation_ids"]) == {30, 31}
    assert not (root / "results/preds").exists()
    loaded, checkpoint = runner().load_temporal_checkpoint(
        result["checkpoint"], model_factory=tiny_factory,
    )
    assert checkpoint["time_config"] == cfg["time"]
    assert checkpoint["architecture"]["mode"] == "both"
    assert loaded.location_encoder[0].bias.requires_grad is False
    from models.geoclip_time.lib.temporal_data import load_temporal_data
    from models.geoclip_time.lib.temporal_ops import predict_temporal
    data = load_temporal_data(root, split_path, time_config=checkpoint["time_config"])
    expected_ids = data.table.loc[data.table.split.eq("val"), "id"].tolist()
    assert log["validation_ids"] == expected_ids
    grid = np.load(root / "geo/grid_1km.npy")
    left = predict_temporal(result["model"], data.features, data.times, grid)
    right = predict_temporal(loaded, data.features, data.times, grid)
    for a, b in zip(left, right):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
    # The checkpoint's explicit preprocessing contract takes priority over any
    # incidental training-config copy, even if a caller has edited that copy.
    checkpoint["config"]["time"] = {**cfg["time"], "year_center": 1850.0}
    torch.save(checkpoint, result["checkpoint"])
    prediction = runner().run_prediction(
        result["checkpoint"], data_root=root, split_csv=split_path,
        splits=("val", "test_in", "test_city"), model_factory=tiny_factory,
    )
    out = pd.read_parquet(prediction["predictions"])
    expected = data.table[data.table.split.isin(["val", "test_in", "test_city"])]
    assert out.id.tolist() == expected.id.tolist()
    np.testing.assert_allclose(out[["lat_pred", "lon_pred"]], right[0][expected.index])
    assert set(out.split) == {"val", "test_in", "test_city"}
    assert json.loads(prediction["metrics"].read_text())["val"]["n"] == 2
    with pytest.raises(FileExistsError):
        runner().run_prediction(result["checkpoint"], data_root=root, split_csv=split_path, model_factory=tiny_factory)


def test_partial_train_cache_requires_explicit_opt_in_and_check_is_read_only(fixture_data):
    root, _, cfg = fixture_data
    ids = np.load(root / "features/clip_vitl14_ids.npy")
    features = np.load(root / "features/clip_vitl14.npy")
    keep = ids != 12
    np.save(root / "features/clip_vitl14_ids.npy", ids[keep])
    np.save(root / "features/clip_vitl14.npy", features[keep])
    with pytest.raises(ValueError, match="partial|missing|coverage"):
        runner().check_training_data(cfg)
    cfg["allow_partial_features"] = True
    audit = runner().check_training_data(cfg)
    assert audit["coverage"]["train"]["missing"] == 1
    assert not (root / "ckpt").exists()
    assert not (root / "results").exists()


@pytest.mark.parametrize("artifact", ["ckpt/tiny.pt", "results/metrics/tiny_train_log.json"])
def test_existing_run_artifacts_are_never_overwritten_by_default(fixture_data, artifact):
    root, _, cfg = fixture_data
    path = root / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("do not replace")
    with pytest.raises(FileExistsError):
        runner().run_training(cfg, model_factory=tiny_factory)
    assert path.read_text() == "do not replace"


def test_cli_help_does_not_create_data_directories(tmp_path):
    project = Path(__file__).resolve().parents[3]
    sentinel = tmp_path / "does-not-exist"
    env = {**os.environ, "P2_DATA": str(sentinel)}
    for script in ("train_time.py", "predict_time.py"):
        result = subprocess.run([sys.executable, str(project / "models/geoclip_time/scripts" / script), "--help"], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    assert not sentinel.exists()


def test_validation_gallery_bases_are_encoded_once_across_query_batches(fixture_data):
    _, _, cfg = fixture_data
    cfg["batch_size"] = 1
    result = runner().run_training(cfg, model_factory=tiny_factory)
    assert result["model"].location_encoder.eval_rows == 7


def test_check_data_cli_never_imports_geoclip_or_writes_outputs(fixture_data, tmp_path):
    root, _, cfg = fixture_data
    import yaml
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    (tmp_path / "geoclip.py").write_text("raise RuntimeError('GeoCLIP must not be imported for data checks')\n")
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "HF_HUB_OFFLINE": "1"}
    script = Path(__file__).resolve().parents[3] / "models/geoclip_time/scripts/train_time.py"
    result = subprocess.run([sys.executable, str(script), "--config", str(config), "--check-data"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["coverage"]["train"]["available"] == 3
    assert not (root / "ckpt").exists()
    assert not (root / "results").exists()


def test_validation_subset_is_deterministic_and_checkpoint_rejects_missing_weights(fixture_data):
    _, _, cfg = fixture_data
    cfg["val_limit"] = 1
    result = runner().run_training(cfg, model_factory=tiny_factory)
    assert result["validation_ids"] == [30]
    state = torch.load(result["checkpoint"], weights_only=True)
    del state["model_state"]["image_mlp.weight"]
    torch.save(state, result["checkpoint"])
    with pytest.raises(RuntimeError, match="Missing key"):
        runner().load_temporal_checkpoint(result["checkpoint"], model_factory=tiny_factory)
