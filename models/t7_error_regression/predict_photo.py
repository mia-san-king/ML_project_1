"""Estimate T4 and T5 errors in kilometres for one raw CLIP ViT-L/14 photo feature."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from lib.confidence_features import GalleryFeatureExtractor
from models.t5_reranker.predict_photo import predict_indexed_photo
from models.t6_router.bundle import DEFAULT_BUNDLE, load_bundle
from models.t6_router.predict_photo import _predict_t4
from models.t7_error_regression import model


def predict_photo(*, feature, t7_model, bundle=DEFAULT_BUNDLE, date_min=None, date_max=None, device="cpu"):
    """Return each frozen expert's coordinates and expected error, plus the chosen prediction."""
    fitted = joblib.load(t7_model)
    files = load_bundle(bundle)
    t4 = _predict_t4(files["t4"], feature, files["border"], date_min, date_max)
    t5 = predict_indexed_photo(model_path=files["t5"], gallery_path=files["t5_gallery"],
                               feature=feature, date_min=date_min, date_max=date_max, device=device)

    gallery = GalleryFeatureExtractor(np.load(files["gallery_latlon"], allow_pickle=False))
    inputs = gallery.transform([[t4["lat_pred"], t4["lon_pred"]]], [t4["has_date"]]).assign(
        T4_lat_pred=t4["lat_pred"], T4_lon_pred=t4["lon_pred"],
        T5_lat_pred=t5["lat_pred"], T5_lon_pred=t5["lon_pred"], t4_was_projected=t4["was_projected"],
        t5_visual_rank=t5["visual_rank"], t5_t3_score=t5["t3_score"], t5_location_cosine=t5["location_cosine"],
    )
    prediction = model.predict(fitted, inputs).iloc[0]
    branches = {
        name: {"lat_pred": expert["lat_pred"], "lon_pred": expert["lon_pred"],
               "expected_error_km": float(prediction[f"{name}_expected_km"])}
        for name, expert in (("T4", t4), ("T5", t5))
    }
    chosen = "T4" if prediction["T7_choose_T4"] else "T5"
    return {"selected_model": chosen, **branches[chosen], "branches": branches}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", type=Path, required=True, help="Raw pre-projection CLIP feature (.npy)")
    parser.add_argument("--t7-model", type=Path, required=True, help="T7.joblib from the expected-error training run")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE, help="Frozen expert bundle manifest (T6.json)")
    parser.add_argument("--date-min", help="Lower endpoint of the capture interval (ISO date or year)")
    parser.add_argument("--date-max", help="Upper endpoint of the capture interval (ISO date or year)")
    parser.add_argument("--device", default="cpu", help="Device for T5 location scoring, e.g. cpu or cuda")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    print(json.dumps(predict_photo(**vars(args)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
