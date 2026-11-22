"""Augmented training configuration and unique-parent retrieval contracts."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch


def module(name):
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, f"{name} is not implemented"
    return importlib.import_module(name)


def references():
    return pd.DataFrame({"id": [-12, 11, 12, -11, 13], "split": ["train"] * 5,
                         "latitude": [47., 46., 47., 46., 48.], "longitude": [7.] * 5,
                         "date_shot_min": [None, "1900", None, "1900", "2000"],
                         "date_shot_max": [None, "1900", None, "1900", "2000"]})


def test_parent_layout_keeps_original_cache_order_and_rejects_corrupt_variants():
    helper = module("scripts.augmentation.parent_retrieval")
    rows = references()
    originals, groups = helper.parent_reference_layout(rows)
    np.testing.assert_array_equal(originals, [1, 2, 4])
    np.testing.assert_array_equal(groups, [1, 0, 1, 0, 2])
    for column, value in [("latitude", 45.), ("date_shot_min", "1950"), ("split", "val")]:
        bad = rows.copy()
        bad.loc[0, column] = value
        with pytest.raises(ValueError):
            helper.parent_reference_layout(bad)
    with pytest.raises(ValueError, match="parent"):
        helper.parent_reference_layout(rows.drop(index=2))


@pytest.mark.parametrize("chunk", [1, 2])
def test_parent_topk_collapses_before_k_and_variant_can_win(chunk):
    helper = module("scripts.augmentation.parent_retrieval")
    # First two references share parent 0: keeping raw top2 then dedup is wrong.
    refs = np.array([[1., 0.], [.99, .1], [0., 1.], [.8, .6], [-1., 0.]])
    groups = np.array([0, 0, 1, 1, 2])
    query = np.array([[2., 0.], [0., 3.], [2., 0.]])
    progress = []
    indices, scores = helper.exact_parent_cosine_topk(query, refs, groups, top_k=2, query_chunk=chunk,
                                                     progress_callback=lambda done, total: progress.append((done, total)))
    np.testing.assert_array_equal(indices, [[0, 1], [1, 0], [0, 1]])
    np.testing.assert_allclose(scores[[0, 2]], [[1., .8], [1., .8]], atol=1e-6)
    assert scores[1, 0] == 1.
    assert progress[-1] == (3, 3)
    assert all(len(np.unique(row)) == 2 for row in indices)


def test_unused_real_splits_are_outside_evaluation_but_can_never_hold_variants():
    helper = module("scripts.augmentation.parent_retrieval")
    rows = references()
    excluded = rows.iloc[[0, 0]].copy()
    excluded["id"] = [81, 82]
    excluded["split"] = ["buffer", "nadir_train"]
    query = excluded.iloc[[0]].assign(id=83, split="test_in")
    all_rows = pd.concat([rows, excluded, query], ignore_index=True)
    assert helper.validate_augmented_rows(all_rows)["evaluation_originals"] == 1
    all_rows.loc[all_rows.id == 81, "id"] = -81
    with pytest.raises(ValueError, match="synthetic"):
        helper.validate_augmented_rows(all_rows)


def test_parent_topk_exact_ties_use_parent_order_and_near_ties_stay_distinct():
    helper = module("scripts.augmentation.parent_retrieval")
    indices, _ = helper.exact_parent_cosine_topk([[1., 0.]], [[1., 0.]] * 6,
                                               [2, 1, 0, 2, 0, 1], top_k=2)
    np.testing.assert_array_equal(indices, [[0, 1]])
    indices, _ = helper.exact_parent_cosine_topk([[1., 0.]], [[1., .001], [1., 0.]], [0, 1], top_k=2)
    np.testing.assert_array_equal(indices, [[1, 0]])


@pytest.mark.parametrize("groups,k", [([0, 2], 1), ([0, -1], 1), ([0., 1.], 1), ([0], 1), ([0, 0], 2)])
def test_parent_topk_rejects_invalid_mapping_or_k(groups, k):
    with pytest.raises(ValueError):
        module("scripts.augmentation.parent_retrieval").exact_parent_cosine_topk(
            [[1., 0.]], [[1., 0.], [0., 1.]], groups, top_k=k)


def test_augmented_config_keeps_original_training_settings_and_isolates_outputs(tmp_path):
    launcher = module("scripts.augmentation.train_models")
    augmented = tmp_path / "augmented"
    for name in ("meta", "features", "geo", "splits"):
        (augmented / name).mkdir(parents=True)
    output = tmp_path / "run"
    cfg0 = launcher.training_config("T0", augmented, output)
    cfg1 = launcher.training_config("T1", augmented, output)
    assert cfg0["model"]["mode"] == "none" and cfg1["model"]["mode"] == "both"
    for cfg in (cfg0, cfg1):
        assert (cfg["epochs"], cfg["seed"], cfg["batch_size"], cfg["n_neg"], cfg["val_limit"]) == (15, 0, 32, 128, 256)
        assert cfg["lr_mlp"] == 1e-4 and cfg["lr_loc"] == 3e-5
        assert cfg["initialization"] == "pretrained"
        assert cfg["split_csv"] == str(augmented / "splits/main.csv.gz")
        assert cfg["allow_partial_features"] is True
    view = launcher.prepare_model_view("T0", augmented, output)
    assert view == output / "T0" / "data"
    assert (view / "features").is_symlink() and (view / "features").resolve() == augmented / "features"
    assert not (augmented / "ckpt").exists()
    assert not (view / "ckpt").is_symlink()
    with pytest.raises(ValueError, match="outside"):
        launcher.training_config("T0", augmented, augmented / "outputs")


def test_cpu_parent_topk_handles_empty_queries_and_rejects_bad_vectors():
    helper = module("scripts.augmentation.parent_retrieval")
    indices, scores = helper.exact_parent_cosine_topk(np.empty((0, 2)), [[1., 0.]], [0], top_k=1)
    assert indices.shape == scores.shape == (0, 1)
    with pytest.raises(ValueError):
        helper.exact_parent_cosine_topk([[0., 0.]], [[1., 0.]], [0], top_k=1)


@pytest.fixture
def augmented_cache(tmp_path):
    root = tmp_path / "augmented"
    for folder in ("features", "meta", "geo", "splits"):
        (root / folder).mkdir(parents=True)
    ids = np.array([21, 11, -12, 12, 13, -11, 24, 22, 23], dtype=np.int64)
    rows = pd.DataFrame({"id": ids, "split": ["val", "train", "train", "train", "train", "train",
                                             "test_region", "test_in", "test_city"]})
    rows["latitude"] = [46., 46., 47., 47., 48., 46., 46., 46., 46.]
    rows["longitude"] = 7.
    rows["date_shot_min"] = ["1900", "1900", "1980", "1980", "1960", "1900", "1900", "1900", "1900"]
    rows["date_shot_max"] = rows.date_shot_min
    rows.drop(columns="split").to_parquet(root / "meta/smapshot.parquet", index=False)
    rows[["id", "split"]].to_csv(root / "splits/main.csv.gz", index=False)
    features = np.zeros((len(ids), 768), dtype=np.float32)
    features[:, :2] = [[0., 1.], [1., 0.], [-1., 0.], [1., .1], [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.]]
    np.save(root / "features/clip_vitl14.npy", features)
    np.save(root / "features/clip_vitl14_ids.npy", ids)
    np.save(root / "geo/grid_1km.npy", np.array([[46., 7.], [47., 7.]], dtype=np.float32))
    return root


def test_preflight_is_read_only_and_rejects_synthetic_queries(augmented_cache, tmp_path):
    launcher = module("scripts.augmentation.train_models")
    output = tmp_path / "outputs"
    audit = launcher.check_inputs(augmented_cache, output)
    assert audit["augmentation"] == {"train_originals": 3, "train_variants": 2, "train_rows": 5,
                                     "parents_with_variant": 2, "evaluation_originals": 4}
    assert not output.exists()
    split = augmented_cache / "splits/main.csv.gz"
    rows = pd.read_csv(split)
    rows.loc[rows.id.eq(-11), "split"] = "val"
    rows.to_csv(split, index=False)
    with pytest.raises(ValueError, match="synthetic"):
        launcher.check_inputs(augmented_cache, output)


def test_complete_parent_retrieval_uses_original_ids_and_validation_only(augmented_cache, tmp_path, monkeypatch):
    from models.geoclip_time.lib.temporal_data import load_temporal_data
    from models.geoclip_time.lib import temporal_training
    from models.geoclip_time.scripts.rerank_references import run_retrieval

    class FrozenProjection(torch.nn.Module):
        mode, embed_dim, time_dim = "none", 512, 3

        def encode_image(self, features, times):
            return torch.nn.functional.normalize(features[:, :512], dim=1)

    split = augmented_cache / "splits/main.csv.gz"
    data = load_temporal_data(augmented_cache, split)
    payload = {"architecture": {"mode": "none"}, "config": {"model": {"mode": "none"},
               "features": "clip_vitl14", "initialization": "pretrained", "allow_partial_features": True},
               "epoch": 1, "history": [{"epoch": 1}], "audit": data.audit, "validation_ids": [21],
               "time_config": data.audit["dates"]["config"]}
    # Substitute only model restoration: real cache loading, grouping, retrieval,
    # val-only gamma selection, winner-ID join and artifact writing remain intact.
    monkeypatch.setattr(temporal_training, "load_temporal_checkpoint",
                        lambda path, device: (FrozenProjection().to(device), payload))
    checkpoint = tmp_path / "fixture-checkpoint.pt"
    checkpoint.write_bytes(b"test-only frozen projection checkpoint")
    result = run_retrieval(data_root=augmented_cache, split_csv=split, checkpoint=checkpoint,
                           output_dir=tmp_path / "T3", parent_grouping=True, threads=1, query_chunk=2)
    pred = pd.read_parquet(tmp_path / "T3/predictions.parquet")
    assert set(pred.id) == {21, 22, 23, 24} and len(pred) == 8
    assert pred.neighbor_id.eq(11).all()  # its variant wins; still report original 11.
    assert pred.lat_pred.eq(46.).all() and pred.lon_pred.eq(7.).all()
    assert result["references"]["n"] == result["top_k_effective"] == 3
    assert result["parent_grouping"]["reference_embedding_count"] == 5
    assert result["selected_gamma"] == 0.
    assert result["test_gamma_search"] is False
    assert result["queries"]["val"]["n"] == 1


def test_launcher_trains_fresh_towers_and_exports_only_original_queries(augmented_cache, tmp_path, monkeypatch):
    from models.geoclip_time.lib import temporal_training
    from models.geoclip_time.tests.test_temporal_training import tiny_factory

    launcher = module("scripts.augmentation.train_models")
    training, prediction = temporal_training.run_training, temporal_training.run_prediction
    initializations = []

    def factory(architecture, *, pretrained):
        initializations.append(pretrained)
        return tiny_factory(architecture, pretrained=pretrained)

    # Substitute the large tower construction only; run all configured 15 epochs.
    monkeypatch.setattr(temporal_training, "run_training",
                        lambda *args, **kwargs: training(*args, **kwargs, model_factory=factory))
    monkeypatch.setattr(temporal_training, "run_prediction",
                        lambda *args, **kwargs: prediction(*args, **kwargs, model_factory=factory))
    output = tmp_path / "outputs"
    trained = launcher.run_action("train", data_root=augmented_cache, output_root=output,
                                  model="T0", threads=1)
    log = json.loads(trained["log"].read_text())
    assert len(log["history"]) == 15 and all(epoch["train_examples"] == 5 for epoch in log["history"])
    assert log["validation_ids"] == [21] and initializations == [True]
    predicted = launcher.run_action("predict", data_root=augmented_cache, output_root=output,
                                    model="T0", threads=1)
    rows = pd.read_parquet(predicted["predictions"])
    assert set(rows.id) == {21, 22, 23, 24}
    assert initializations == [True, False]
    assert not (augmented_cache / "ckpt").exists() and not (augmented_cache / "results").exists()
    assert trained["checkpoint"].is_relative_to(output / "T0")


def test_check_cli_runs_from_outside_repo_without_creating_outputs(augmented_cache, tmp_path):
    launcher = module("scripts.augmentation.train_models")
    output = tmp_path / "outputs"
    result = subprocess.run([sys.executable, launcher.__file__, "check", "--data-root", str(augmented_cache),
                             "--output-root", str(output)], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["augmentation"]["train_variants"] == 2
    assert not output.exists()


def test_t4_launcher_fits_all_augmented_rows_and_writes_isolated_original_queries(augmented_cache, tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    border = gpd.GeoDataFrame(geometry=[box(2400000, 1000000, 2900000, 1400000)], crs=2056)
    border.to_file(augmented_cache / "geo/ne_admin0.gpkg")
    output = tmp_path / "outputs"
    report = module("scripts.augmentation.train_models").run_action(
        "t4", data_root=augmented_cache, output_root=output, threads=1)
    assert report["train_n"] == 5
    assert set(pd.read_parquet(output / "T4/predictions.parquet").id) == {21, 22, 23, 24}
    assert not (augmented_cache / "ckpt").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_parent_topk_matches_cpu_for_well_separated_scores():
    helper = module("scripts.augmentation.parent_retrieval")
    query, refs, groups = [[1., 0.], [0., 1.]], [[1., 0.], [.8, .6], [0., 1.]], [0, 0, 1]
    cpu = helper.exact_parent_cosine_topk(query, refs, groups, top_k=2)
    gpu = helper.exact_parent_cosine_topk(query, refs, groups, top_k=2, device="cuda")
    np.testing.assert_array_equal(cpu[0], gpu[0])
    np.testing.assert_allclose(cpu[1], gpu[1], atol=1e-6)
