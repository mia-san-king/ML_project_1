from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from models.t6_router import region_experts as expert


def table():
    ids = [7, -7, 12, 31, -31, 53]
    return pd.DataFrame({"id": ids, "feat_idx": range(len(ids)), "split": ["train", "train", "train", "query", "ignore", "val"],
                         "latitude": [46.0, 46.0, 47.0, 47.1, 47.1, 46.3],
                         "longitude": [7.0, 7.0, 8.0, 8.1, 8.1, 7.7], "block": range(6),
                         "date_shot_min": [1900] * 6, "date_shot_max": [1910] * 6})


def test_gallery_only_contains_allowed_parents_and_query_is_real():
    data = table()
    ref, parents, groups, queries = expert._partition(data)
    assert ref.id.tolist() == [7, -7, 12]
    assert parents.id.tolist() == [7, 12]
    assert groups.tolist() == [0, 0, 1]
    assert queries.id.tolist() == [31]
    data.loc[data.id == -31, "split"] = "train"
    with pytest.raises(ValueError, match="parent"):
        expert._partition(data)
    data.loc[data.id == -31, "split"] = "query"
    with pytest.raises(ValueError, match="positive"):
        expert._partition(data)


def test_region_t5_predicts_training_parent_and_records_its_scores():
    data = table()
    ref, parents, groups, queries = expert._partition(data)
    raw = np.zeros((len(data), 2), dtype=np.float32)
    raw[:, 0] = 1.
    raw[2:4] = [0., 1.]
    archive = SimpleNamespace(table=data, times=np.zeros((len(data), 3), np.float32), features=raw)

    class T0:
        embed_dim = 2
        time_dim = 3
        def encode_image(self, features, times):
            return features

    class T1:
        mode = "both"
        def encode_image(self, features, times):
            return features
        def encode_location_base(self, gps):
            return torch.stack([47. - gps[:, 0], gps[:, 0] - 46.], dim=1)
        def location_time_adapter(self, bases, times):
            return bases

    pred = expert._predict_t5(archive, ref, parents, groups, queries, T0(), T1(), device="cpu", top_k=2)
    assert pred.T5_neighbor_id.tolist() == [12]
    assert pred.t5_visual_rank.tolist() == [1]
    assert pred.T5_lat_pred.tolist() == [47.]
    np.testing.assert_allclose(pred[["t5_t3_score", "t5_location_cosine"]], [[1., 1.]])


def test_fold_retrains_experts_and_reuses_only_same_inputs(tmp_path, monkeypatch):
    import geopandas as gpd
    import yaml
    from shapely.geometry import box
    from models.geoclip_time.lib import temporal_training
    from models.geoclip_time.tests.test_temporal_training import tiny_factory
    from models.t6_router import region_towers

    data = tmp_path / "augmented"
    for directory in ("meta", "geo", "features"):
        (data / directory).mkdir(parents=True)
    ids = np.array([67, *range(1, 61), -9, 61, 62, 63, 64, 65, -64, 66])
    rng = np.random.default_rng(7)
    np.save(data / "features/clip_vitl14_ids.npy", ids)
    np.save(data / "features/clip_vitl14.npy", rng.normal(size=(len(ids), 768)).astype(np.float16))
    source = pd.DataFrame({"id": ids, "latitude": 46. + np.abs(ids) / 100.,
                           "longitude": 7. + np.abs(ids) / 100., "date_shot_min": 1900, "date_shot_max": 1910})
    source.sample(frac=1, random_state=1).to_parquet(data / "meta/smapshot.parquet", index=False)
    np.save(data / "geo/grid_1km.npy", source[["latitude", "longitude"]].to_numpy()[:25])
    roles = source[["id"]].assign(split="train", block=1, region="")
    roles.loc[roles.id.isin([61, 62, 63]), "split"] = "val"
    roles.loc[roles.id.isin([64, 65, 66]), ["split", "region"]] = ["query", "final"]
    roles.loc[roles.id.isin([-64, 67]), "split"] = "ignore"
    split = tmp_path / "split.csv"
    roles.sample(frac=1, random_state=2).to_csv(split, index=False)
    border = tmp_path / "data/local_runs/t6_aug8192/bundle_v2/ne_admin0.gpkg"
    border.parent.mkdir(parents=True)
    gpd.GeoDataFrame({"geometry": [box(5., 45., 10., 48.)]}, crs=4326).to_file(border)
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    for tower, path in region_towers.CONFIGS.items():
        cfg = {"epochs": 1, "n_neg": 4, "batch_size": 16, "model": {"mode": "none" if tower == "T0" else "both"}}
        (cfgdir / path).write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(region_towers, "CONFIG_DIR", cfgdir)
    monkeypatch.setattr(region_towers, "run_training", lambda *a, **kw:
                        temporal_training.run_training(*a, **kw, model_factory=tiny_factory))
    monkeypatch.setattr(expert, "load_temporal_checkpoint", lambda *a, **kw:
                        temporal_training.load_temporal_checkpoint(*a, **kw, model_factory=tiny_factory))
    out = tmp_path / "run"
    result = expert.run_experts(tmp_path, data, split, out, device="cpu")
    frame = pd.read_parquet(result)
    assert frame.id.tolist() == [64, 65, 66]
    assert frame.region.eq("final").all()
    assert frame.T5_neighbor_id.between(1, 60).all()
    assert frame.T5_error_km.notna().all() and frame.T4_error_km.notna().all()
    assert (out / "experts/T5.pt").exists() and (out / "experts/T4.pt").exists()
    np.testing.assert_array_equal(np.load(out / "experts/t5_gallery.npz")["ids"], np.arange(1, 61))
    assert not (data / "ckpt").exists()
    assert expert.run_experts(tmp_path, data, split, out, device="cpu") == result
    roles.loc[roles.id.eq(63), "split"] = "ignore"
    roles.to_csv(split, index=False)
    with pytest.raises(ValueError, match="inputs changed"):
        expert.run_experts(tmp_path, data, split, out, device="cpu")
