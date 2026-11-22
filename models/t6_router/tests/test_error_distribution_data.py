"""All four models must evaluate the same held-out photos and canonical GPS."""
import numpy as np
import pandas as pd
import pytest

from models.t6_router import error_distribution_data as comparison


@pytest.mark.parametrize("fault", [None, "misassigned", "bad_distance", "modified_file"])
def test_four_way_test_alignment(tmp_path, monkeypatch, fault):
    splits = ["train", "val", "test_in", "test_city", "test_region"]
    base = pd.DataFrame({"id": [11, 4, 8, 7, 3], "split": splits, "block": splits,
                         "latitude": range(45, 50), "longitude": range(5, 10)})
    frames = []
    for model in ("T1", "T3", "T4", "T5"):
        pred = base.iloc[1:].rename(columns={"latitude": "lat_pred", "longitude": "lon_pred"}).copy()
        pred["model"], pred["p_within_25km"], pred["error_km"] = model, 0.63, 0.
        pred["confidence_partition"] = np.where(pred.split == "val", "fit", "test")
        if model == "T3" and fault == "misassigned":
            pred.loc[pred.split == "test_city", "block"] = "test_in"
        if model == "T1" and fault == "bad_distance":
            pred.loc[pred.split == "test_in", "error_km"] = 50.
        frames.append(pred)

    routes = {
        "split": "data/splits/main.csv.gz", "metadata": "data/t1_original/meta/smapshot.parquet",
        "ids": "data/t1_original/features/clip_vitl14_ids.npy",
        "predictions": "data/local_runs/confidence_models/predictions.parquet",
    }
    paths = {k: tmp_path / v for k, v in routes.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    base[["id", "split", "block"]].to_csv(paths["split"], index=False)
    base[["id", "latitude", "longitude"]].to_parquet(paths["metadata"])
    np.save(paths["ids"], base.id.to_numpy())
    pd.concat(frames).sample(frac=1, random_state=42).to_parquet(paths["predictions"])
    hashes = pd.DataFrame([{"root": "workspace", "path": routes[k], "sha256": comparison.sha256(v)}
                           for k, v in paths.items()])
    archive = tmp_path / "assets.tsv"
    hashes.to_csv(archive, sep="\t", index=False)
    monkeypatch.setattr(comparison, "ARCHIVE", archive)
    monkeypatch.setattr(comparison, "EXPECTED", dict.fromkeys(comparison.TEST_SPLITS, 1))

    if fault == "modified_file":
        np.save(paths["ids"], base.id.to_numpy()[::-1])
    if fault is not None:
        messages = {"misassigned": "wrong test partition or geographic block",
                    "bad_distance": "errors disagree", "modified_file": "differs from the frozen experiment"}
        with pytest.raises(ValueError, match=messages[fault]):
            comparison.load_test_comparison(paths["predictions"])
        return
    paired = comparison.load_test_comparison(paths["predictions"])
    assert paired.id.tolist() == [3, 7, 8]
    assert set(paired.split) == set(comparison.TEST_SPLITS)
    for model in ("T1", "T3", "T4", "T5"):
        np.testing.assert_array_equal(paired[f"{model}_lat_pred"], paired.latitude)
        np.testing.assert_array_equal(paired[f"{model}_error_km"], np.zeros(len(paired)))
