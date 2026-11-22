"""Fit the matched geographic-sampling T4 experiment and compare spatial blocks."""
import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from experiment_paths import VOLUME
from time import perf_counter
from unittest.mock import patch

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.clip_regression.scripts import train
from models.geoclip_time.lib.temporal_data import TemporalData, load_temporal_data
from models.geoclip_time.scripts.rerank_references import _sha256
from select_t4_geo4096 import digest

RUNS = ROOT / "data/local_runs"
OUT = RUNS / "t4_geo4096"
PREV = RUNS / "t3_t4_8192"
DATA = PREV / "t4_source/original_view"
ARTIFACTS = VOLUME / "aug8192/t1_dataset"
SIZE = 4096
SPLITS = {"val": 29529, "test_in": 47593, "test_city": 3377, "test_region": 13039}
NAMES = ("baseline", "random", "sparse")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    dest = path.with_suffix(path.suffix + ".tmp")
    dest.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    dest.replace(path)


def write_npy(path, array):
    if path.exists():
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), array)
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def original_data():
    size_digest = read_json(ROOT / "report/experiments/2026-11-18-geoclip-time/run/input_manifest.json")
    digests = {Path(i["path"]).name: i["sha256"] for i in size_digest["inputs"]}
    for relative in ("features/clip_vitl14.npy", "features/clip_vitl14_ids.npy", "meta/smapshot.parquet", "splits/main.csv.gz"):
        if _sha256(DATA / relative) != digests[Path(relative).name]:
            raise ValueError(f"Original input mismatch: {relative}")
    data = load_temporal_data(DATA, DATA / "splits/main.csv.gz")
    if not (data.table.id > 0).all() or data.table.split.eq("train").sum() != 176308:
        raise ValueError("Expected only original rows and 176308 training images")
    return data


def random_inputs(data):
    eligible = np.sort(data.table.loc[data.table.split.eq("train"), "id"].to_numpy(dtype=np.int64))
    ids = np.random.default_rng(0).permutation(eligible)[:SIZE]
    manifest = read_json(ARTIFACTS / "build_manifest.json")
    for key in ("features/clip_vitl14.npy", "features/clip_vitl14_ids.npy"):
        if _sha256(ARTIFACTS / key) != manifest["artifact_sha256"][key]:
            raise ValueError("Previously verified augmentation cache mismatch")
    all_ids = pd.Index(np.load(ARTIFACTS / "features/clip_vitl14_ids.npy", allow_pickle=False))
    positions = all_ids.get_indexer(-ids)
    if np.any(positions < 0):
        raise ValueError("Random parent absent from published 8192")
    features = np.load(ARTIFACTS / "features/clip_vitl14.npy", mmap_mode="r", allow_pickle=False)[positions]
    write_npy(OUT / "random_ids.npy", ids)
    write_npy(OUT / "random_features.npy", features)


def augment(data, ids_path, feats_path):
    parents = np.load(ids_path, allow_pickle=False)
    features = np.load(feats_path, allow_pickle=False)
    if parents.dtype != np.int64 or parents.shape != (SIZE,) or (parents <= 0).any() or len(np.unique(parents)) != SIZE:
        raise ValueError("Require 4096 distinct int64 positive parent IDs")
    if features.shape != (SIZE, 768) or features.dtype != np.float16 or not np.isfinite(features).all():
        raise ValueError("Require finite raw FP16 CLIP features of shape [4096,768]")
    real_rows = data.table
    positions = pd.Index(real_rows.id).get_indexer(parents)
    if (positions < 0).any() or not real_rows.iloc[positions].split.eq("train").all():
        raise ValueError("Missing parent or non-training parent")
    if not np.array_equal(real_rows.feat_idx, np.arange(len(real_rows))):
        raise ValueError("Date, metadata and feature indexes must be aligned")
    synth = real_rows.iloc[positions].copy()
    synth["id"] = -parents
    synth["feat_idx"] = np.arange(len(real_rows), len(real_rows) + SIZE)
    audit = copy.deepcopy(data.audit)
    audit["coverage"]["train"]["expected"] += SIZE
    audit["coverage"]["train"]["available"] += SIZE
    for name in ("metadata_rows", "feature_rows", "split_rows"):
        audit[name] += SIZE
    return TemporalData(features=np.concatenate((data.features, features)),
                        table=pd.concat((real_rows, synth), ignore_index=True),
                        times=np.concatenate((data.times, data.times[positions])), audit=audit)


def fit(arm, ids_path, feats_path):
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    data = original_data()
    if arm == "random":
        random_inputs(data)
    sources = {}
    protocol = read_json(OUT / "protocol.json")
    if arm != "baseline":
        ids_path = ids_path or OUT / f"{arm}_ids.npy"
        feats_path = feats_path or OUT / f"{arm}_features.npy"
        data = augment(data, ids_path, feats_path)
        sources = {label: {"path": str(path), "sha256": _sha256(path)}
                   for label, path in (("parents", ids_path), ("features", feats_path))}
        accepted = np.load(ids_path, allow_pickle=False)
        if arm == "random" and digest(accepted) != protocol["random_ids_sha256"]:
            raise ValueError("Random parent order differs from the frozen protocol")
        if arm == "sparse":
            priority = np.load(OUT / "sparse_parents.npy", allow_pickle=False)
            ranks = pd.Index(priority).get_indexer(accepted)
            if digest(priority) != protocol["sparse_priority_ids_sha256"] or (ranks < 0).any() or (np.diff(ranks) <= 0).any():
                raise ValueError("Sparse parents differ from the frozen priority order")
            receipt = read_json(OUT / "sparse_features.json")
            if sources["parents"]["sha256"] != receipt["ids_sha256"] or sources["features"]["sha256"] != receipt["features_sha256"]:
                raise ValueError("Sparse input files differ from the audited encoder output")
    dest = OUT / "models" / arm
    if dest.exists():
        external = read_json(dest / "fit_receipt.json")
        if external["augmented_inputs"] != sources or _sha256(dest / "predictions.parquet") != external["predictions_sha256"]:
            raise ValueError("Existing run differs from input files or has changed predictions")
        print("Already verified", arm, flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with tempfile.TemporaryDirectory(dir=dest.parent) as temporary:
        stage = Path(temporary) / arm
        # Supply the standard T4 runner with an in-memory TemporalData view.
        # Every synthetic metadata row and date encoding is copied from its positive parent.
        with patch.object(train, "load_temporal_data", return_value=data):
            result = train.fit_and_predict(DATA, DATA / "splits/main.csv.gz", stage)
        elapsed = perf_counter() - started
        n_train = 176308 + (SIZE if arm != "baseline" else 0)
        if result["train_n"] != n_train or result["design_rank"] != 772:
            raise ValueError("Unexpected training count or design rank")
        pred = pd.read_parquet(stage / "predictions.parquet").set_index("id", verify_integrity=True)
        if pred.split.value_counts().to_dict() != SPLITS or not (pred.index > 0).all():
            raise ValueError("Unexpected real query set")
        if arm != "sparse":
            key = "diffusion_0" if arm == "baseline" else "diffusion_4096"
            previous = read_json(PREV / "t4_augmentation_curve.json")["results"][key]
            for split, group in pred.groupby("split"):
                if (group.error_km <= 25).sum() != previous[split]["within25"]:
                    raise ValueError(f"Old T4 hit count failed reproduction: {arm} {split}")
                np.testing.assert_allclose(group.error_km.median(), previous[split]["median_km"], rtol=0, atol=1e-7)
        write_json(stage / "fit_receipt.json",
                   {"arm": arm, "real_rows": 176308, "augmented_rows": n_train - 176308, "train_rows": n_train,
                    "date_gps_source": "indexed positive train parents", "augmented_inputs": sources,
                    "solver": result["solver"], "torch": str(torch.__version__), "device": "cpu",
                    "threads": torch.get_num_threads(), "fit_and_predict_seconds": elapsed,
                    "predictions_sha256": _sha256(stage / "predictions.parquet"),
                    "weights_sha256": _sha256(stage / "T4.pt")})
        stage.rename(dest)
    print("Completed", arm, dest, flush=True)


def clustered_intervals(counts, net):
    # Pairs are always resampled together; sample whole distinct geographic blocks.
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(counts), size=(10000, len(counts)))
    micro = 100 * net[draws].sum(axis=1) / counts[draws].sum(axis=1)
    macro = np.mean(100 * (net / counts)[draws], axis=1)
    return {"photo_weighted_delta_pp_ci95": np.quantile(micro, [.025, .975]).tolist(),
            "equal_block_delta_pp_ci95": np.quantile(macro, [.025, .975]).tolist(),
            "loo_micro_delta_pp_range": [float(x) for x in [np.min(100*(net.sum()-net)/(counts.sum()-counts)),
                                                            np.max(100*(net.sum()-net)/(counts.sum()-counts))]]}


def compare():
    data = original_data()
    truth = data.table.loc[data.table.split.isin(SPLITS), ["id", "split", "block", "latitude", "longitude"]]
    truth = truth.set_index("id", verify_integrity=True).sort_index()
    train_blocks = set(data.table.loc[data.table.split.eq("train"), "block"])
    if truth.block.isna().any() or train_blocks.intersection(truth.loc[truth.split.isin(["val", "test_in"]), "block"]):
        raise ValueError("Canonical validation and domestic test blocks must be held out")
    protocol = read_json(OUT / "protocol.json")
    for split, group in truth.groupby("split"):
        planned = protocol["endpoints"][split]
        if (digest(group.index) != planned["sorted_query_ids_sha256"] or
                len(group) != planned["n"] or group.block.nunique() != planned["blocks"]):
            raise ValueError("Evaluation IDs/blocks differ from the frozen protocol")
    provenance = {}
    for arm in NAMES:
        dest = OUT / "models" / arm
        receipt = read_json(dest / "fit_receipt.json")
        path = dest / "predictions.parquet"
        if _sha256(path) != receipt["predictions_sha256"]:
            raise ValueError("Changed predictions")
        provenance[arm] = receipt
        frame = pd.read_parquet(path).set_index("id", verify_integrity=True).sort_index()
        pd.testing.assert_frame_equal(frame[["split", "latitude", "longitude"]], truth[["split", "latitude", "longitude"]])
        errors = train._distance_errors(truth, frame[["lat_pred", "lon_pred"]].to_numpy())
        np.testing.assert_allclose(errors, frame.error_km, rtol=0, atol=1e-7)
        truth[arm] = errors <= 25
        truth[f"{arm}_km"] = errors
    truth["gained"] = ~truth["random"] & truth["sparse"]
    truth["lost"] = truth["random"] & ~truth["sparse"]
    rows = truth.groupby(["split", "block"], sort=True)
    by_block = rows[list(NAMES) + ["gained", "lost"]].sum()
    by_block["n"] = rows.size()
    by_block["net"] = by_block["sparse"] - by_block["random"]
    by_block["sparse_minus_random_pp"] = 100*by_block.net / by_block.n
    blockout = OUT / "paired_blocks.parquet"
    tmp = blockout.with_suffix(".tmp")
    by_block.reset_index().to_parquet(tmp, index=False)
    tmp.replace(blockout)
    outputs = {}
    for split, total in SPLITS.items():
        group = truth.loc[truth.split.eq(split)]
        blocks = by_block.loc[split]
        if len(group) != total:
            raise ValueError("Unexpected evaluation size")
        stats = {arm: {"hits": int(group[arm].sum()), "photo_weighted_pct": float(100*group[arm].mean()),
                       "equal_block_pct": float((100*blocks[arm]/blocks.n).mean()),
                       "median_error_km": float(group[f"{arm}_km"].median())} for arm in NAMES}
        pairs = {"gained": int(group.gained.sum()), "lost": int(group.lost.sum()),
                 "net": int(blocks.net.sum()), "benefiting_blocks": int((blocks.net > 0).sum()),
                 "worsening_blocks": int((blocks.net < 0).sum()), "tied_blocks": int((blocks.net == 0).sum()),
                 "photo_weighted_delta_pp": float(100*blocks.net.sum()/blocks.n.sum()),
                 "equal_block_delta_pp": float((100*blocks.net/blocks.n).mean())}
        if split in ("val", "test_in"):
            pairs.update(clustered_intervals(blocks.n.to_numpy(), blocks.net.to_numpy()))
        outputs[split] = {"n": total, "blocks": len(blocks), "arms": stats, "sparse_minus_random": pairs}
        print(split, json.dumps(outputs[split], ensure_ascii=True), flush=True)
    protocol = OUT / "protocol.json"
    report = {"primary": "val photo-weighted sparse minus random 25 km hit rate, percentage points",
              "threshold_km": 25., "bootstrap": "paired percentile, seed 0, 10000 resamples of whole blocks per split",
              "fits": provenance, "splits": outputs, "paired_blocks_sha256": _sha256(blockout)}
    if protocol.exists():
        report["protocol_sha256"] = _sha256(protocol)
    write_json(OUT / "comparison.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=[*NAMES, "compare"])
    parser.add_argument("--parents", type=Path)
    parser.add_argument("--features", type=Path)
    args = parser.parse_args()
    if args.action == "compare":
        compare()
    else:
        fit(args.action, args.parents, args.features)
