"""Route a single raw CLIP ViT-L/14 feature through the paired T4 and T5 models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from pyproj import Transformer

from lib.confidence import GeoConfidence
from lib.confidence_features import GalleryFeatureExtractor
from models.clip_regression.lib.regression import LinearTimeRegressor, clip_to_switzerland
from models.geoclip_time.lib.temporal_data import encode_date_intervals
from models.t5_reranker.predict_photo import predict_indexed_photo
from models.t6_router.bundle import load_bundle
from models.t6_router.router import choose_t4


def _predict_t4(checkpoint, feature_path, border_path, date_min, date_max):
    spec = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (spec["model"], spec["feature_name"], spec["feature_dim"], spec["output_crs"]) != (
        "T4", "clip_vitl14", 768, "EPSG:2056"
    ):
        raise ValueError("Expected a T4 raw CLIP ViT-L/14 checkpoint with LV95 outputs")
    feature = np.load(feature_path, allow_pickle=False)
    if feature.shape not in {(768,), (1, 768)} or feature.dtype.kind not in "fiu" or not np.isfinite(feature).all():
        raise ValueError("Expected a finite real CLIP ViT-L/14 feature of shape [768] or [1, 768]")
    interval = pd.DataFrame({"date_shot_min": [date_min], "date_shot_max": [date_max]})
    times, _ = encode_date_intervals(interval, **spec["time_config"])
    model = LinearTimeRegressor(spec["feature_dim"]).double().eval()
    model.load_state_dict(spec["model_state"])
    with torch.inference_mode():
        offset = model(torch.from_numpy(feature.reshape(1, -1)).double(), torch.from_numpy(times).double()).numpy()
    raw = np.asarray(spec["origin_m"]) + offset * spec["scale_m"]
    border = gpd.read_file(border_path).to_crs(2056).geometry.union_all()
    xy = clip_to_switzerland(raw, border)
    east, north = xy[0]
    lon, lat = Transformer.from_crs(2056, 4326, always_xy=True).transform(east, north)
    return {"lat_pred": float(lat), "lon_pred": float(lon),
            "has_date": bool(times[0, 2]), "was_projected": bool(np.any(raw != xy))}


def predict_photo(*, model_path, feature, date_min=None, date_max=None, device="cpu", gate_path=None):
    """Return the selected GPS, each branch's 25 km probability, and its own GPS."""
    files = load_bundle(model_path)
    t4_cal = GeoConfidence.load(files["t4_confidence"])
    t5_cal = GeoConfidence.load(files["t5_confidence"])
    if t4_cal.threshold_km != 25.0 or t5_cal.threshold_km != 25.0:
        raise ValueError("T6 expects two 25 km confidence estimators")
    t4 = _predict_t4(files["t4"], feature, files["border"], date_min, date_max)
    t5 = predict_indexed_photo(model_path=files["t5"], gallery_path=files["t5_gallery"], feature=feature,
                               date_min=date_min, date_max=date_max, device=device)

    gallery = GalleryFeatureExtractor(np.load(files["gallery_latlon"], allow_pickle=False))
    t4_features = gallery.transform([[t4["lat_pred"], t4["lon_pred"]]], [t4["has_date"]])
    t4_features["t4_was_projected"] = t4["was_projected"]
    t5_features = pd.DataFrame([{"t5_visual_rank": t5["visual_rank"], "t5_t3_score": t5["t3_score"],
                                 "t5_location_cosine": t5["location_cosine"]}])
    p4, p5 = t4_cal.predict_proba(t4_features), t5_cal.predict_proba(t5_features)
    gate_info = {}
    if gate_path is None:
        use_t4 = choose_t4(p4, p5)[0]
    else:
        import joblib

        inputs = pd.DataFrame([{"T4_lat_pred": t4["lat_pred"], "T4_lon_pred": t4["lon_pred"],
                                "T5_lat_pred": t5["lat_pred"], "T5_lon_pred": t5["lon_pred"],
                                "T4_p25": p4[0], "T5_p25": p5[0]}])
        probability = float(joblib.load(gate_path).predict_t4_probability(inputs)[0])
        use_t4 = probability > 0.5
        gate_info["gate_p_T4_given_one_hit"] = probability
    name = "T4" if use_t4 else "T5"
    branches = {"T4": {**t4, "p_within_25km": float(p4[0])}, "T5": {**t5, "p_within_25km": float(p5[0])}}
    selected = branches[name]
    return {"selected_model": name, "lat_pred": selected["lat_pred"], "lon_pred": selected["lon_pred"],
            "selected_branch_p_within_25km": selected["p_within_25km"], "branches": branches, **gate_info}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="model_path", type=Path, required=True, help="T6.json manifest")
    parser.add_argument("--feature", type=Path, required=True, help="One raw pre-projection CLIP feature (.npy)")
    parser.add_argument("--date-min", help="Lower endpoint of the capture interval (ISO date or year)")
    parser.add_argument("--date-max", help="Upper endpoint of the capture interval (ISO date or year)")
    parser.add_argument("--device", default="cpu", help="T5 location scoring device, e.g. cuda or cpu")
    parser.add_argument("--gate", dest="gate_path", type=Path, help="Joint six-input gate.joblib from train_geographic_gate")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(predict_photo(**vars(args)), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
