"""Assemble inference-time features and paired evaluation labels for local augmented models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.confidence_features import EARTH_KM, SINGLE_FEATURES, GalleryFeatureExtractor
from models.clip_regression.scripts.compare_t1 import _provenance
from models.geoclip_time.lib.temporal_data import encode_date_intervals
from models.t5_reranker.evaluate import GPS, PREDICTED, original_rows, read_predictions

RUNS = ROOT / "data/local_runs"
NATIVE_COLUMNS = {
    "T1": [],
    "T3": ["visual_score", "year_penalty", "score"],
    "T4": ["was_projected"],
    "T5": ["visual_rank", "t3_score", "location_cosine", "score"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True, help="Augmented T1 predictions parquet")
    parser.add_argument("--t3", type=Path, default=RUNS / "t3_t4_8192/augmented/T3/predictions.parquet")
    parser.add_argument("--t4", type=Path, default=RUNS / "t3_t4_8192/augmented/T4/predictions.parquet")
    parser.add_argument("--t5", type=Path, default=RUNS / "t5_aug8192/results/predictions.parquet")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/t1_original")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/splits/main.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=RUNS / "confidence_models")
    args = parser.parse_args()

    truth, train, sources = original_rows(args.data_root, args.splits)
    date_columns = ["id", "date_shot_min", "date_shot_max"]
    dates = pd.read_parquet(sources["metadata"], columns=date_columns).set_index("id", verify_integrity=True)
    encoded, _ = encode_date_intervals(dates.loc[truth.index])
    known = encoded[:, 2].astype(bool)
    paths = dict(T1=args.t1, T3=args.t3, T4=args.t4, T5=args.t5)
    gallery = GalleryFeatureExtractor(train[list(GPS)].to_numpy(dtype=float))
    rows = []
    for model, path in paths.items():
        frame, errors = read_predictions(path, model, truth, train)
        predicted = frame[list(PREDICTED)].to_numpy(dtype=np.float64)
        if model == "T4" and not np.array_equal(known, frame.has_date):
            raise ValueError("Query date mask failed to match saved T4 date mask")
        features = gallery.transform(predicted, known, workers=4).set_axis(frame.index)
        native = frame[NATIVE_COLUMNS[model]].add_prefix(model.lower() + "_")
        sample = truth[["split", "block"]].copy()
        sample["model"] = model
        sample["error_km"] = errors
        sample[list(PREDICTED)] = predicted
        if model in ("T3", "T5") and not features.nearest_train_km.eq(0).all():
            raise ValueError(f"{model} reported a retrieval outside the real training gallery")
        rows.append(sample.join(features).join(native))

    data = pd.concat(rows).reset_index()
    required = ["id", "model", "split", "block", "error_km", *PREDICTED, *SINGLE_FEATURES]
    if data[required].isna().any().any() or data.duplicated(["model", "id"]).any():
        raise ValueError("Incomplete features, evaluation keys or labels")
    data = data[required + [col for col in data if col not in required]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "inputs.parquet"
    data.to_parquet(output, index=False)
    manifest = {
        "rows": len(data), "real_train_gallery_photos": len(train),
        "queries_by_split": truth.split.value_counts().to_dict(),
        "single_features": SINGLE_FEATURES,
        "native_features": {m: [m.lower() + "_" + c for c in NATIVE_COLUMNS[m]] for m in paths},
        "definitions": {
            "nearest_train_km": "Spherical distance from predicted point to nearest real training photo.",
            "log1p_train_5km,log1p_train_25km": "Natural log of one plus the count of real training photos within radius of the predicted point. Co-located photos count individually.",
            "has_date": "Both calendar endpoints of the query date interval passed the GeoCLIP time encoder checks.",
            "error_km": "Evaluation-only label from canonical true GPS. Spatial block and split are evaluation metadata.",
            "earth_radius_km": EARTH_KM,
        }, "inputs": {name: _provenance(path) for name, path in {**sources, **paths}.items()},
        "output": _provenance(output),
    }
    (args.output_dir / "inputs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(data):,} query-model predictions written to {output}")


if __name__ == "__main__":
    main()
