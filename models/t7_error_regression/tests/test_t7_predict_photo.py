import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from models.t7_error_regression import model, predict_photo as inference


@pytest.mark.parametrize("km4, km5, dates, winner", [
    (8., 24., ("1930", "1939"), "T4"),
    (16., 8., (None, None), "T5"),
    (8., 8., ("1900", None), "T5"),
])
def test_single_photo_uses_batch_inputs_and_returns_both_estimates(tmp_path, monkeypatch, km4, km5, dates, winner):
    saved = tmp_path / "T7.joblib"
    gallery = tmp_path / "gallery.npy"
    np.save(gallery, [[46., 7.], [46.1, 7.], [47., 8.]])
    experts = {"t4": "t4.pt", "border": "ch.gpkg", "t5": "t5.pt",
               "t5_gallery": "index.npz", "gallery_latlon": gallery}
    monkeypatch.setattr(inference, "load_bundle", lambda path: experts if path == "T6.json" else {})

    def t4(checkpoint, feature, border, low, high):
        assert (checkpoint, feature, border, low, high) == ("t4.pt", "photo.npy", "ch.gpkg", *dates)
        return {"lat_pred": 46., "lon_pred": 7., "was_projected": True, "has_date": any(dates)}

    def t5(**kwargs):
        assert kwargs == {"model_path": "t5.pt", "gallery_path": "index.npz", "feature": "photo.npy",
                          "date_min": dates[0], "date_max": dates[1], "device": "cuda"}
        return {"lat_pred": 47., "lon_pred": 8., "visual_rank": 11, "t3_score": .37, "location_cosine": -.2}

    monkeypatch.setattr(inference, "_predict_t4", t4)
    monkeypatch.setattr(inference, "predict_indexed_photo", t5)
    regressors = [DummyRegressor(strategy="constant", constant=km).fit([[0]], [km]) for km in (km4, km5)]
    joblib.dump(model.ExpectedErrorModel(*regressors), saved)
    real_predict = model.predict

    def check_model_inputs(fitted, frame):
        assert len(frame) == 1
        assert frame.iloc[0].to_dict() == {
            "T4_lat_pred": 46., "T4_lon_pred": 7., "T5_lat_pred": 47., "T5_lon_pred": 8.,
            "t4_was_projected": True, "has_date": any(dates), "nearest_train_km": 0.,
            "log1p_train_5km": np.log1p(1), "log1p_train_25km": np.log1p(2),
            "t5_visual_rank": 11, "t5_t3_score": .37, "t5_location_cosine": -.2,
        }
        return real_predict(fitted, frame)

    monkeypatch.setattr(model, "predict", check_model_inputs)
    result = inference.predict_photo(feature="photo.npy", bundle="T6.json", t7_model=saved,
                                     date_min=dates[0], date_max=dates[1], device="cuda")
    assert result["branches"] == {"T4": {"lat_pred": 46., "lon_pred": 7., "expected_error_km": km4},
                                  "T5": {"lat_pred": 47., "lon_pred": 8., "expected_error_km": km5}}
    assert result == {"selected_model": winner, **result["branches"][winner], "branches": result["branches"]}


def test_cli_passes_feature_interval_and_model_paths_as_json(monkeypatch, capsys):
    def predict(**kwargs):
        assert kwargs == {"feature": Path("photo.npy"), "bundle": Path("T6.json"), "t7_model": Path("T7.joblib"),
                          "date_min": "1930", "date_max": "1939", "device": "cpu"}
        return {"selected_model": "T4"}

    monkeypatch.setattr(inference, "predict_photo", predict)
    inference.main(["--feature", "photo.npy", "--bundle", "T6.json", "--t7-model", "T7.joblib",
                    "--date-min", "1930", "--date-max", "1939", "--device", "cpu"])
    assert json.loads(capsys.readouterr().out) == {"selected_model": "T4"}
