"""Package the frozen T4/T5 experts, their gallery, calibrators and the learned T6 gate."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from lib.confidence import GeoConfidence
from models.geoclip_time.scripts.rerank_references import _sha256


LEGACY_FILES = {
    "t4": "T4.pt", "t5": "T5.pt",
    "t4_confidence": "T4.joblib", "t5_confidence": "T5.joblib",
    "gallery_latlon": "train_latlon.npy", "t5_gallery": "t5_gallery.npz",
    "border": "ne_admin0.gpkg",
}
FILES = {**LEGACY_FILES, "gate": "gate.joblib"}
LEGACY_RULE = "T4 if p4_within_25km > p5_within_25km else T5"
RULE = "T4 if gate_p_T4_given_one_hit > 0.5 else T5"

# Frozen deployment pair from the 2026-11-22 asset manifest and geographic_gate.json.
FROZEN_SHA256 = {
    "t4": "2330dae7214bf705d0d168b8377b1b99846a250d6e20a439de56e31678fa3ada",
    "t5": "9893ec3dd6419378853eadf9fa8856e7ab0a56da6789d255d9f41371c890e286",
    "t4_confidence": "25fdda8baf0bcd50f686ea4c9bce842c16df9eee8c0cb7fd6c8354222b8fa2c7",
    "t5_confidence": "64107184a1340f7afa7402a71cb96c2711b20536e5e99abd14326abfcae6e734",
    "gate": "9c083b4d4d1ae2b3a1ddd2dae413f3090d6111285322d41b67911a2ffb24bfd3",
}
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "local_runs"
DEFAULT_BUNDLE = RUNS / "t6_aug8192" / "bundle_v2" / "T6.json"


def _check_frozen(entries):
    for key, sha in FROZEN_SHA256.items():
        if entries[key]["sha256"] != sha:
            raise ValueError(f"Unexpected weights for the published six-input T6: {key}")


def _read_bundle(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("model") != "T6":
        raise ValueError("Expected a T6 package")
    version = manifest.get("version")
    if version == 1 and manifest.get("rule") == LEGACY_RULE:
        names = LEGACY_FILES
    elif version == 2 and manifest.get("rule") == RULE:
        names = FILES
    else:
        raise ValueError("Unknown T6 package version or selection rule")
    entries = manifest["files"]
    if set(entries) != set(names):
        raise ValueError("Incomplete T6 asset list")
    if version == 2:
        _check_frozen(entries)
    assets = {}
    for key, name in names.items():
        asset = path.parent / name
        if entries[key]["name"] != name or _sha256(asset) != entries[key]["sha256"]:
            raise ValueError(f"T6 asset differs from manifest: {key}")
        assets[key] = asset
    return manifest, assets


def load_bundle(path):
    """Read and verify the assets of a published or archived T6 package."""
    return _read_bundle(Path(path).resolve())[1]


def _publish(staging, output, metadata):
    entries = {key: {"name": name, "sha256": _sha256(staging / name)} for key, name in FILES.items()}
    _check_frozen(entries)
    manifest = {**metadata, "model": "T6", "version": 2, "rule": RULE, "files": entries}
    (staging / "T6.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staging.rename(output)
    return output / "T6.json"


def _new_directory(output):
    if output.exists():
        raise FileExistsError(f"Choose a new T6 bundle directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)


def upgrade_bundle(*, from_bundle, gate, output):
    """Publish a separate v2 bundle, reusing immutable local files when possible."""
    _new_directory(output)
    source, assets = _read_bundle(Path(from_bundle).resolve())
    if source["version"] != 1:
        raise ValueError("--from-bundle expects an archived T6 version 1 manifest")
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        for key, asset in assets.items():
            target = staging / FILES[key]
            try:
                os.link(asset, target)
            except OSError:
                shutil.copyfile(asset, target)
        shutil.copyfile(gate, staging / FILES["gate"])
        metadata = {key: value for key, value in source.items() if key not in ("files", "rule", "version", "model")}
        return _publish(staging, output, metadata)


def build_bundle(*, output, gate, t4_dir, t5_dir, confidence_dir, data_root, split_csv, border):
    from models.t5_reranker.predict_photo import prepare_gallery
    from scripts.confidence.evaluate import FEATURES

    _new_directory(output)
    t5_run = json.loads((t5_dir / "run.json").read_text(encoding="utf-8"))
    confidence = json.loads((confidence_dir / "comparison.json").read_text(encoding="utf-8"))
    inputs = json.loads((confidence_dir / "inputs_manifest.json").read_text(encoding="utf-8"))
    if confidence["input"]["sha256"] != inputs["output"]["sha256"]:
        raise ValueError("Confidence fit and feature manifest refer to different inputs")
    weights = {"t4": t4_dir / "T4.pt", "t5": t5_dir / "T5.pt", "border": border, "gate": gate}
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
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / "bundle"
        staging.mkdir()
        for key, source in weights.items():
            shutil.copyfile(source, staging / FILES[key])
        np.savez(staging / FILES["t5_gallery"], **gallery)
        np.save(staging / FILES["gallery_latlon"], gallery["gps"], allow_pickle=False)
        return _publish(staging, output, {
            "real_training_photos": len(gallery["ids"]), "indexed_vectors": len(gallery["vectors"]),
            "confidence_sklearn_version": confidence["sklearn_version"],
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paired = RUNS / "t3_t4_8192"
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-bundle", type=Path, help="Reuse an existing v1 T6.json and its assets")
    source.add_argument("--data-root", type=Path, help="Build the T5 gallery from the augmented feature cache")
    parser.add_argument("--gate", type=Path, required=True, help="Published six-input gate.joblib")
    parser.add_argument("--t4-dir", type=Path, default=paired / "augmented/T4")
    parser.add_argument("--t5-dir", type=Path, default=RUNS / "t5_aug8192/results")
    parser.add_argument("--confidence-dir", type=Path, default=RUNS / "confidence_models")
    parser.add_argument("--split-csv", type=Path, help="Defaults to splits/main.csv.gz in --data-root")
    parser.add_argument("--border", type=Path, default=paired / "t4_source/ne_admin0.gpkg")
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE.parent)
    args = parser.parse_args()
    if args.from_bundle is not None:
        manifest = upgrade_bundle(from_bundle=args.from_bundle, gate=args.gate, output=args.output)
    else:
        del args.from_bundle
        args.split_csv = args.split_csv or args.data_root / "splits/main.csv.gz"
        torch.set_num_threads(4)
        manifest = build_bundle(**vars(args))
    print(manifest)


if __name__ == "__main__":
    main()
