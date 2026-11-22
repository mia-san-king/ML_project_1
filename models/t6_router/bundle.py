"""Package frozen T4/T5 locators, their 25 km confidence models and the train-only gallery."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from lib.confidence import GeoConfidence
from models.geoclip_time.scripts.rerank_references import _sha256


FILES = {
    "t4": "T4.pt", "t5": "T5.pt",
    "t4_confidence": "T4.joblib", "t5_confidence": "T5.joblib",
    "gallery_latlon": "train_latlon.npy", "t5_gallery": "t5_gallery.npz",
    "border": "ne_admin0.gpkg",
}
RULE = "T4 if p4_within_25km > p5_within_25km else T5"
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "local_runs"


def load_bundle(path):
    """Return verified local asset paths for a format-1 T6 package."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("model"), manifest.get("version"), manifest.get("rule")) != ("T6", 1, RULE):
        raise ValueError("Expected a T6 version 1 package with the fixed 25 km selection rule")
    entries = manifest["files"]
    if set(entries) != set(FILES):
        raise ValueError("Incomplete T6 asset list")
    assets = {}
    for key, name in FILES.items():
        asset = path.parent / name
        if entries[key]["name"] != name or _sha256(asset) != entries[key]["sha256"]:
            raise ValueError(f"T6 asset differs from manifest: {key}")
        assets[key] = asset
    return assets


def build_bundle(*, output, t4_dir, t5_dir, confidence_dir, data_root, split_csv, border):
    from models.t5_reranker.predict_photo import prepare_gallery
    from scripts.confidence.evaluate import FEATURES

    if output.exists():
        raise FileExistsError(f"Choose a new T6 bundle directory: {output}")
    t5_run = json.loads((t5_dir / "run.json").read_text(encoding="utf-8"))
    confidence = json.loads((confidence_dir / "comparison.json").read_text(encoding="utf-8"))
    inputs = json.loads((confidence_dir / "inputs_manifest.json").read_text(encoding="utf-8"))
    if confidence["input"]["sha256"] != inputs["output"]["sha256"]:
        raise ValueError("Confidence fit and feature manifest refer to different inputs")
    weights = {"t4": t4_dir / "T4.pt", "t5": t5_dir / "T5.pt", "border": border}
    if _sha256(weights["t5"]) != t5_run["checkpoint_sha256"]:
        raise ValueError("T5 weights differ from the recorded model")
    for name, directory in (("T4", t4_dir), ("T5", t5_dir)):
        if _sha256(directory / "predictions.parquet") != inputs["inputs"][name]["sha256"]:
            raise ValueError(f"Confidence inputs were generated using different {name} predictions")
        cal_path = confidence_dir / f"{name}.joblib"
        if _sha256(cal_path) != confidence["artifacts"][name]["sha256"]:
            raise ValueError(f"Confidence weights differ from the {name} calibration report")
        cal = GeoConfidence.load(cal_path)
        if cal.threshold_km != 25 or list(cal.feature_names) != FEATURES[name]:
            raise ValueError(f"Unexpected {name} confidence radius or input schema")
        weights[name.lower() + "_confidence"] = cal_path

    t5 = torch.load(weights["t5"], weights_only=True, map_location="cpu")
    gallery = prepare_gallery(t5, data_root, split_csv)
    if len(gallery["gps"]) != inputs["real_train_gallery_photos"]:
        raise ValueError("T5 and confidence models have different real training gallery counts")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        for key, source in weights.items():
            shutil.copyfile(source, staging / FILES[key])
        np.savez(staging / FILES["t5_gallery"], **gallery)
        np.save(staging / FILES["gallery_latlon"], gallery["gps"], allow_pickle=False)
        files = {key: {"name": name, "sha256": _sha256(staging / name)} for key, name in FILES.items()}
        manifest = {"model": "T6", "version": 1, "rule": RULE, "files": files,
                    "real_training_photos": len(gallery["ids"]), "indexed_vectors": len(gallery["vectors"]),
                    "confidence_sklearn_version": confidence["sklearn_version"]}
        (staging / "T6.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    return output / "T6.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paired = RUNS / "t3_t4_8192"
    parser.add_argument("--t4-dir", type=Path, default=paired / "augmented/T4")
    parser.add_argument("--t5-dir", type=Path, default=RUNS / "t5_aug8192/results")
    parser.add_argument("--confidence-dir", type=Path, default=RUNS / "confidence_models")
    parser.add_argument("--data-root", type=Path, required=True, help="Augmented feature cache used by T5")
    parser.add_argument("--split-csv", type=Path, help="Defaults to splits/main.csv.gz in --data-root")
    parser.add_argument("--border", type=Path, default=paired / "t4_source/ne_admin0.gpkg")
    parser.add_argument("--output", type=Path, default=RUNS / "t6_aug8192/bundle")
    args = parser.parse_args()
    args.split_csv = args.split_csv or args.data_root / "splits/main.csv.gz"
    torch.set_num_threads(4)
    print(build_bundle(**vars(args)))


if __name__ == "__main__":
    main()
