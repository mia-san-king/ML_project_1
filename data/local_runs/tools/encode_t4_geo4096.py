"""Assemble sparse-arm frozen CLIP vectors after the selected images are published."""
import json
import shutil
from pathlib import Path
from experiment_paths import VOLUME

from augment_train import ORIGINAL, encode_new, encoder_identity
from run_original_t1 import ROOT, RUNS, read_json
from select_t4_geo4096 import describe, digest
from scripts.augmentation.build_dataset import _sha256, _atomic_json
import numpy as np
import pandas as pd

WORK = RUNS / "t4_geo4096"
PRIOR = VOLUME / "aug8192"
CLIP = VOLUME / "public-assets/clip"
STORE = VOLUME / "t4_geo4096"


def main():
    protocol = read_json(WORK / "protocol.json")
    order = np.load(WORK / "sparse_parents.npy")
    if digest(order) != protocol["sparse_priority_ids_sha256"]:
        raise ValueError("Protocol and parent order differ")
    chosen = pd.read_parquet(STORE / "selected.parquet").sort_values("rank")
    ids = chosen.parent_id.to_numpy(dtype=np.int64)
    if len(ids) != 4096 or len(set(ids)) != len(ids):
        raise ValueError("Need 4,096 distinct parents")
    expected_order = pd.Index(order).get_indexer(ids)
    if (expected_order < 0).any() or (np.diff(expected_order) <= 0).any():
        raise ValueError("Unexpected selection sequence")
    dataset = PRIOR / "t1_dataset"
    manifest = read_json(dataset / "build_manifest.json")
    for file in ("features/clip_vitl14.npy","features/clip_vitl14_ids.npy"):
        if _sha256(dataset / file) != manifest["artifact_sha256"][file]:
            raise ValueError(f"Archived feature dataset changed at {file}")
    feature_ids = pd.Index(np.load(dataset / "features/clip_vitl14_ids.npy"))
    features = np.load(dataset / "features/clip_vitl14.npy", mmap_mode="r")
    real_ids = np.load(ORIGINAL / "features/clip_vitl14_ids.npy")
    split = pd.read_csv(ROOT / "data/splits/main.csv.gz",usecols=["id","split","block"]).set_index("id",verify_integrity=True)
    train = split.loc[np.intersect1d(real_ids,split.index[split.split.eq("train")])]
    if not np.isin(ids,train.index).all():
        raise ValueError("Ineligible parent selected")
    cached = feature_ids.get_indexer(-ids)
    if not np.array_equal(cached >= 0, chosen.source.eq("cached_8192").to_numpy()) or not chosen.source.isin(["cached_8192","new_sdxl"]).all():
        raise ValueError("Invalid reuse of archived features")
    staging = WORK / "encoding"
    staging.mkdir(exist_ok=True)
    calibration = PRIOR / "clip_build_calibration.json"
    identity = encoder_identity(CLIP)
    if identity != read_json(calibration)["encoder"]:
        raise ValueError("Encoder differs from archived 8,192 run")
    shutil.copyfile(calibration, staging / "clip_preflight.json")
    inputs = []
    for row in chosen.loc[chosen.source.eq("new_sdxl")].itertuples():
        photo, variant = Path(row.parent_path), Path(row.image_path)
        if _sha256(photo) != row.parent_sha256 or _sha256(variant) != row.image_sha256:
            raise ValueError(f"Image content changed at {row.parent_id}")
        quality = json.loads(row.quality_json)
        if not quality["passed"]:
            raise ValueError(f"QA failed at {row.parent_id}")
        inputs.append({"parent_id":int(row.parent_id), "image_sha256":row.image_sha256, "original_file":photo, "output_file":variant})
    new_vectors, encoder = encode_new(inputs, feature_ids.to_numpy(), features, CLIP, staging, 16)
    vectors = np.empty((len(ids),features.shape[1]), dtype="<f2")
    for i,(photo,position) in enumerate(zip(ids,cached,strict=True)):
        vectors[i] = features[position] if position >= 0 else new_vectors[int(photo)]
    if not np.isfinite(vectors).all() or not (np.linalg.norm(vectors.astype(np.float32),axis=1)>0).all():
        raise ValueError("Invalid assembled feature values")
    for label, array in (("sparse_ids",ids),("sparse_features",vectors)):
        out = WORK / (label + ".npy")
        tmp = WORK / (label + ".partial.npy")
        np.save(tmp,array)
        tmp.replace(out)
    report = {"n":len(ids), "new_sdxl":len(inputs), "used_cached":int((cached>=0).sum()),
              "replacements":int(sum(expected_order >= 4096)), "selection":describe(ids, train),
              "selected_parquet_sha256":_sha256(STORE / "selected.parquet"),"encoder":encoder,
              "ids_sha256":_sha256(WORK/"sparse_ids.npy"),"features_sha256":_sha256(WORK/"sparse_features.npy")}
    _atomic_json(WORK / "sparse_features.json",report)
    print({k:v for k,v in report.items() if k != "encoder"},flush=True)


if __name__ == "__main__":
    main()
