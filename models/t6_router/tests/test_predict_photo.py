import numpy as np
import pytest

from models.t6_router import predict_photo as inference


@pytest.mark.parametrize(("p4", "p5", "expected"), [(0.6, 0.4, "T4"), (0.2, 0.7, "T5"), (0.5, 0.5, "T5")])
def test_photo_uses_own_branch_evidence_and_returns_its_coordinates(tmp_path, monkeypatch, p4, p5, expected):
    gallery = tmp_path / "gallery.npy"
    np.save(gallery, [[46., 7.], [46., 7.]])
    files = dict(t4="t4.pt", t5="t5.pt", t4_confidence="t4.joblib", t5_confidence="t5.joblib",
                 gallery_latlon=gallery, border="border.gpkg", t5_gallery="t5_gallery.npz")
    monkeypatch.setattr(inference, "load_bundle", lambda path: files)
    monkeypatch.setattr(inference, "_predict_t4", lambda *args: dict(lat_pred=46., lon_pred=7.,
                                                                     has_date=False, was_projected=True))
    monkeypatch.setattr(inference, "predict_indexed_photo", lambda **kwargs: dict(lat_pred=47., lon_pred=8.,
                                                                                 visual_rank=12, t3_score=0.36,
                                                                                 location_cosine=-0.2))

    class CheckInputs:
        threshold_km = 25.

        def __init__(self, filename):
            self.filename = filename

        def predict_proba(self, frame):
            row = frame.iloc[0].to_dict()
            if self.filename == "t4.joblib":
                assert row == {"nearest_train_km": 0., "log1p_train_5km": np.log1p(2),
                               "log1p_train_25km": np.log1p(2), "has_date": False, "t4_was_projected": True}
                return np.array([p4])
            assert row == {"t5_visual_rank": 12, "t5_t3_score": 0.36, "t5_location_cosine": -0.2}
            return np.array([p5])

    monkeypatch.setattr(inference.GeoConfidence, "load", CheckInputs)
    result = inference.predict_photo(model_path="T6.json", feature="x.npy")
    assert result["selected_model"] == expected
    assert (result["lat_pred"], result["lon_pred"]) == ((46., 7.) if expected == "T4" else (47., 8.))
    assert result["branches"]["T4"]["p_within_25km"] == p4
    assert result["branches"]["T5"]["p_within_25km"] == p5
    assert result["selected_branch_p_within_25km"] == max(p4, p5)
