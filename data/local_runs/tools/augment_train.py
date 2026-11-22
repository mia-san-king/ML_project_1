"""Build a verified 8,192-variant CLIP cache and compare a full T1 fit to the local original."""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from experiment_paths import VOLUME
from time import perf_counter

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import portalocker
import yaml

from run_original_t1 import ARCHIVE, CONFIG, ROOT, RUNS, SPLIT, read_json, verify
from compare_clip_variants import merged_digest

sys.path.insert(0, str(ROOT))
from scripts.augmentation.build_dataset import (
    FrozenClipEncoder, _atomic_json, _evaluation_audit, _sha256, validate_parent_features,
)
from models.geoclip_time.lib.temporal_training import EVAL_SPLITS, _distance_errors, _summary

OLD_COUNT = 512
NEW_COUNT = 7680
ORIGINAL = ROOT / "data/t1_original"
PRIOR = ROOT / "report/experiments/2026-11-20-diffusion-augmentation"
RAW = RUNS / "clip_variants/variant_rows.bin"
ORIGINAL_CONFIG = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
BASE_NAME = ORIGINAL_CONFIG["run_name"]
NAME = "T1_aug8192_s0"
DATA_FILES = ("features/clip_vitl14.npy", "features/clip_vitl14_ids.npy", "meta/smapshot.parquet",
              "splits/main.csv.gz", "geo/grid_1km.npy")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_original_inputs():
    for entry in read_json(ARCHIVE / "run/input_manifest.json")["inputs"]:
        relative = Path(*entry["path"].split("/")[-2:])
        verify(SPLIT if relative == Path("splits/main.csv.gz") else ORIGINAL / relative, entry)
    features = np.load(ORIGINAL / DATA_FILES[0], mmap_mode="r", allow_pickle=False)
    ids = np.load(ORIGINAL / DATA_FILES[1], allow_pickle=False)
    prior = read_json(PRIOR / "augmented/build_manifest.json")
    old_ids = np.asarray(prior["synthetic_ids"], dtype=ids.dtype)
    old_vectors = np.fromfile(RAW, dtype="<f2").reshape(OLD_COUNT, 768)
    merged_digest(ids, old_ids, DATA_FILES[1], prior)
    merged_digest(features, old_vectors, DATA_FILES[0], prior)
    return features, ids, -old_ids, old_vectors


def accepted_samples(store, images, expected, old_ids, generation):
    summary = read_json(store / "generation_summary.json")
    if not summary.get("complete") or summary["accepted"] != NEW_COUNT:
        raise ValueError(f"Need {NEW_COUNT} accepted new images and a complete generation summary.")
    previous = set(int(x) for x in old_ids)
    if previous != set(expected[:OLD_COUNT]):
        raise ValueError("Prior 512-image subset disagrees with the original sorted-train seed-0 permutation.")
    with sqlite3.connect((store / "records.sqlite").as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(row) for row in con.execute("SELECT * FROM samples ORDER BY selection_rank")]
    if not rows or [row["selection_rank"] for row in rows] != list(range(OLD_COUNT, OLD_COUNT + len(rows))):
        raise ValueError("Generation ranks are incomplete or duplicated.")
    accepted = []
    for row in rows:
        rank, parent = row["selection_rank"], row["parent_id"]
        if parent != expected[rank] or row["synthetic_id"] != -parent:
            raise ValueError(f"Parent identity differs from canonical selection rank {rank}.")
        if row["status"] in {"rejected", "source_unavailable"}:
            continue
        if row["status"] != "accepted":
            raise ValueError(f"Unfinished generation for parent {parent}: {row['status']}")
        if (parent in previous or row["model_sha256"] != generation["model_sha256"]
                or row["model_revision"] != generation["model_revision"]):
            raise ValueError(f"Duplicate parent or model mismatch at {parent}.")
        if (row["parent_split"], row["is_synthetic"], row["is_diffusion"],
                row["source"], row["generator"]) != ("train", 1, 1, "diffusion", "sdxl_img2img"):
            raise ValueError(f"Invalid synthetic provenance for parent {parent}.")
        if not json.loads(row["quality_json"])["passed"]:
            raise ValueError(f"Generation QA failed for accepted parent {parent}.")
        previous.add(parent)
        parent_path = images / f"{parent // 1000:04d}" / f"{parent}.jpg"
        output_path = store / "images" / f"{parent}.png"
        if (Path(row["parent_path"]).resolve() != parent_path.resolve()
                or (store / row["image_path"]).resolve() != output_path.resolve()):
            raise ValueError(f"Unexpected generation paths for {parent}.")
        if _sha256(parent_path) != row["parent_sha256"] or _sha256(output_path) != row["image_sha256"]:
            raise ValueError(f"Image content changed after generation for parent {parent}.")
        row.update(original_file=parent_path, output_file=output_path)
        accepted.append(row)
    if len(accepted) != NEW_COUNT:
        raise ValueError(f"Found {len(accepted)} accepted new images.")
    return accepted


def encoder_identity(clip):
    runtime = {k: version(k) for k in ("torch", "transformers", "geoclip", "pillow", "numpy")}
    configs = {"config.json", "preprocessor_config.json", "processor_config.json", "image_processor_config.json",
               "tokenizer_config.json", "tokenizer.json", "vocab.json", "special_tokens_map.json", "merges.txt",
               "model.safetensors.index.json", "pytorch_model.bin.index.json"}
    weights = {p.relative_to(clip).as_posix(): _sha256(p) for p in sorted(clip.iterdir())
               if p.is_file() and (p.name in configs or p.suffix.lower() in {".bin", ".safetensors", ".model"})}
    if not any(p.endswith((".safetensors", ".bin")) for p in weights):
        raise FileNotFoundError(f"No CLIP checkpoint found in {clip}")
    return {"runtime": runtime, "weight_hashes": weights, "dtype": "raw_fp16_pre_mlp"}


def prepare_encoder(clip, ids, features, samples, device, report_path, preflight=None):
    from transformers import AutoProcessor

    identity = encoder_identity(clip)
    # Reconfirm the preflight-selected processor on the actual production parents.
    modes = ["default"]
    confirmed_preflight = None
    if preflight and preflight.exists():
        prior = read_json(preflight)
        if prior.get("encoder") == identity and prior.get("selected_processor") is not None:
            confirmed_preflight = prior["selected_processor"]
            modes = [confirmed_preflight]
    parent_ids = [s["parent_id"] for s in samples]
    index = pd.Index(ids).get_indexer(parent_ids)
    if np.any(index < 0):
        raise ValueError("Missing real parent features for CLIP calibration.")
    paths = [s["original_file"] for s in samples]
    report = {"encoder":identity, "device":device, "parent_ids":list(map(int, parent_ids)),
              "source_files":[{"path":str(p.resolve()), "sha256":_sha256(p)} for p in paths],
              "minimum_cosine":0.999, "maximum_norm_relative_error":0.02,
              "preflight_selected_processor":confirmed_preflight,
              "attempts":[], "selected_processor":None}
    encoder = FrozenClipEncoder(clip, device=device, expected_transformers=identity["runtime"]["transformers"])
    for mode in modes:
        if mode != "default":
            encoder.image_processor = AutoProcessor.from_pretrained(
                str(clip), local_files_only=True, use_fast=mode == "fast")
        component = getattr(encoder.image_processor, "image_processor", encoder.image_processor)
        processor = type(component).__name__
        attempt = {"mode":mode, "implementation":processor}
        decoded = encoder.encode(paths)
        try:
            attempt["metrics"] = validate_parent_features(features[index], decoded, parent_ids)
            attempt["passed"] = True
        except ValueError as error:
            attempt.update(passed=False, error=str(error))
        report["attempts"].append(attempt)
        if attempt["passed"]:
            report["selected_processor"] = mode
            break
        if mode == "default" and confirmed_preflight is None:
            modes.append("slow" if processor.endswith("Fast") else "fast")
    _atomic_json(report_path, report)
    print(json.dumps({"calibration_report":str(report_path), "attempts":report["attempts"],
                      "selected_processor":report["selected_processor"]}), flush=True)
    if report["selected_processor"] is None:
        raise ValueError(f"CLIP parent calibration failed. See {report_path}.")
    return encoder, {**identity, "processor_mode":mode, "processor_class":processor}, report


def calibrate(args):
    features, ids, _, _ = load_original_inputs()
    splits = pd.read_csv(SPLIT, usecols=["id","split"])
    eligible = np.intersect1d(ids, splits.loc[splits.split.eq("train"), "id"])
    parents = np.random.default_rng(0).permutation(eligible)
    samples = []
    for photo in parents[:OLD_COUNT + NEW_COUNT]:
        path = args.images / f"{photo // 1000:04d}" / f"{photo}.jpg"
        if path.is_file():
            samples.append({"parent_id":int(photo), "original_file":path})
        if len(samples) == 16:
            break
    if len(samples) != 16:
        raise FileNotFoundError(f"Found {len(samples)}/16 staged real photos under {args.images}.")
    prepare_encoder(args.clip, ids, features, samples, args.device, args.store / "clip_preflight.json")


def encode_new(rows, ids, features, clip, store, batch_size):
    sampled = [rows[i] for i in np.linspace(0, len(rows) - 1, min(16, len(rows)), dtype=int)]
    encoder, identity, calibration = prepare_encoder(clip, ids, features, sampled, "cuda",
                                                     store / "clip_build_calibration.json",
                                                     preflight=store / "clip_preflight.json")
    identity_key = digest(identity)
    result = {}
    with sqlite3.connect(store / "clip_progress.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS vectors(key TEXT PRIMARY KEY, value BLOB NOT NULL, sha256 TEXT NOT NULL)")
        pending = []
        for row in rows:
            key = digest([identity_key, row["parent_id"], row["image_sha256"]])
            cached = db.execute("SELECT value,sha256 FROM vectors WHERE key=?", (key,)).fetchone()
            if cached:
                value, checksum = cached
                if len(value) != 1536 or hashlib.sha256(value).hexdigest() != checksum:
                    raise ValueError(f"Invalid cached vector for {row['parent_id']}.")
                result[row["parent_id"]] = np.frombuffer(value, dtype="<f2")
            else:
                pending.append((key, row))
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            encoded = encoder.encode([row["output_file"] for _, row in batch]).astype("<f2")
            if not np.isfinite(encoded).all() or not np.all(np.any(encoded != 0, axis=1)):
                raise ValueError("Cannot cache nonfinite or identically zero CLIP features.")
            for (key, row), vector in zip(batch, encoded, strict=True):
                value = vector.tobytes()
                db.execute("INSERT INTO vectors VALUES (?,?,?)", (key, value, hashlib.sha256(value).hexdigest()))
                result[row["parent_id"]] = vector.copy()
            db.commit()
            if start % (batch_size * 16) == 0:
                print(f"Encoded {len(result)}/{len(rows)} new variants.", flush=True)
    return result, {**identity, "calibration": calibration}


def append_tables(metadata, splits, ids, parents):
    train = splits.loc[splits.split.eq("train"), "id"]
    if not (len(np.unique(parents)) == len(parents) and np.all(parents > 0)
            and np.isin(parents, ids).all() and np.isin(parents, train).all()):
        raise ValueError("All synthetic parents must be distinct original training images with cached features.")
    source = metadata.set_index("id", verify_integrity=True)
    if (source.index <= 0).any() or (ids <= 0).any():
        raise ValueError("The source contains synthetic IDs.")
    labels = [key for key in ("latitude", "longitude", "date_shot_min", "date_shot_max", "view_type")
              if key in metadata]
    synthetic = source.loc[parents, labels].copy()
    synthetic["id"] = -parents
    synthetic["parent_id"] = pd.array(parents, dtype="Int64")
    synthetic["is_synthetic"] = True
    synthetic["is_diffusion"] = True
    metadata = metadata.assign(parent_id=pd.array([pd.NA] * len(metadata), dtype="Int64"),
                               is_synthetic=False, is_diffusion=False)
    joined = pd.concat([metadata, synthetic], ignore_index=True)
    inherited = splits.set_index("id", verify_integrity=True).loc[parents].copy()
    inherited["id"] = -parents
    if "random_split" in inherited:
        inherited["random_split"] = "train"
    augmented_split = pd.concat([splits, inherited[splits.columns]], ignore_index=True)
    augmented_ids = np.concatenate([ids, -parents])
    audit = _evaluation_audit(ids, augmented_ids, splits, augmented_split, seed=0, limit=256)
    return joined, augmented_split, augmented_ids, audit


def build(args):
    source, ids, old_ids, tail = load_original_inputs()
    splits = pd.read_csv(SPLIT)
    eligible = np.intersect1d(ids, splits.loc[splits.split.eq("train"), "id"])
    expected = np.random.default_rng(0).permutation(eligible)
    prior_generation = read_json(PRIOR / "synthetic_store/generation_manifest.json")
    generation = read_json(args.store / "generation_manifest.json")
    for key in ("model_revision", "model_sha256", "parameters", "input_sha256"):
        if generation[key] != prior_generation[key]:
            raise ValueError(f"Generator {key} differs from the original 512-image SDXL experiment.")
    rows = accepted_samples(args.store, args.images, expected, old_ids, generation)
    snapshot = digest([[r["parent_id"], r["parent_sha256"], r["image_sha256"]] for r in rows])
    dataset = args.store / "t1_dataset"
    if dataset.exists():
        manifest = verify_dataset(dataset)
        if manifest["generation_snapshot"] != snapshot:
            raise ValueError("Dataset already exists with different inputs.")
        return dataset
    encoded, encoder_info = encode_new(rows, ids, source, args.clip, args.store, args.batch_size)
    encoded.update(zip(map(int, old_ids), tail, strict=True))
    parents = np.array(sorted(encoded), dtype=np.int64)
    metadata = pd.read_parquet(ORIGINAL / DATA_FILES[2])
    meta, split, augmented_ids, audit = append_tables(metadata, splits, ids, parents)
    remote_log = read_json(ARCHIVE / "metrics" / f"{BASE_NAME}_train_log.json")
    if audit["validation_ids"] != remote_log["validation_ids"]:
        raise ValueError("Checkpoint selection IDs differ from the original T1 experiment.")
    stage = args.store / "t1_dataset.staging"
    for filename in DATA_FILES:
        (stage / filename).parent.mkdir(exist_ok=True, parents=True)
    out = np.lib.format.open_memmap(stage / DATA_FILES[0], mode="w+", dtype=source.dtype,
                                    shape=(len(augmented_ids), 768))
    for start in range(0, len(ids), 8192):
        out[start:min(start + 8192, len(ids))] = source[start:start + 8192]
    out[len(ids):] = np.stack([encoded[i] for i in parents])
    out.flush()
    del out
    np.save(stage / DATA_FILES[1], augmented_ids)
    meta.to_parquet(stage / DATA_FILES[2], index=False)
    split.to_csv(stage / DATA_FILES[3], index=False, compression={"method": "gzip", "mtime": 0})
    shutil.copyfile(ORIGINAL / DATA_FILES[4], stage / DATA_FILES[4])
    manifest = {"archived_parents": len(old_ids), "new_parents": len(rows), "parent_count": len(parents),
                "source_sha256": _sha256(ARCHIVE / "run/input_manifest.json"),
                "original_512_tail_sha256": _sha256(RAW), "generation_snapshot": snapshot,
                "generation_manifest_sha256": _sha256(args.store / "generation_manifest.json"),
                "encoder": encoder_info, "validation_and_test": audit, "eligible_train": len(eligible),
                "artifact_sha256": {path: _sha256(stage / path) for path in DATA_FILES}}
    _atomic_json(stage / "build_manifest.json", manifest)
    stage.rename(dataset)
    print(f"Published {dataset}", flush=True)
    return dataset


def verify_dataset(dataset):
    manifest = read_json(dataset / "build_manifest.json")
    if (manifest["archived_parents"], manifest["new_parents"], manifest["parent_count"]) != (OLD_COUNT, NEW_COUNT, OLD_COUNT + NEW_COUNT):
        raise ValueError("Wrong augmentation counts in published dataset.")
    for file in DATA_FILES:
        if _sha256(dataset / file) != manifest["artifact_sha256"][file]:
            raise ValueError(f"Published data changed: {file}")
    return manifest


def paired_results(before, after, truth, splits):
    first = before.set_index("id", verify_integrity=True).sort_index()
    last = after.set_index("id", verify_integrity=True).sort_index()
    if not first.index.equals(last.index) or (first.index <= 0).any() or not first.split.equals(last.split):
        raise ValueError("The two predictions do not refer to identical original query IDs/splits.")
    truth = truth.set_index("id", verify_integrity=True).loc[first.index]
    regions = splits.set_index("id", verify_integrity=True).loc[first.index]
    if not first.split.equals(regions.split) or set(first.split) != set(EVAL_SPLITS):
        raise ValueError("Prediction split labels disagree with source.")
    old = _distance_errors(truth, first[["lat_pred", "lon_pred"]])
    new = _distance_errors(truth, last[["lat_pred", "lon_pred"]])
    if not np.isfinite([old, new]).all():
        raise ValueError("Nonfinite geographic error.")
    result = {}
    for label in EVAL_SPLITS:
        keep = regions.split.eq(label).to_numpy()
        a, b = old[keep], new[keep]
        result[label] = {"n": len(a), "original": _summary(a), "augmented": _summary(b),
                         "median_gain_km": float(np.median(a) - np.median(b)),
                         "acc25_gain_pp": float(100 * ((b <= 25).mean() - (a <= 25).mean())),
                         "gained_hits": int(((b <= 25) & (a > 25)).sum()),
                         "lost_hits": int(((a <= 25) & (b > 25)).sum())}
        if label == "test_in":
            blocks = regions.loc[keep, "block"]
            if blocks.isna().any() or blocks.nunique() < 2:
                raise ValueError("Need complete geographic blocks for paired uncertainty.")
            members = [np.flatnonzero(blocks.to_numpy() == value) for value in np.sort(blocks.unique())]
            rng = np.random.default_rng(0)
            changes = []
            for _ in range(1000):
                pos = np.concatenate([members[k] for k in rng.integers(len(members), size=len(members))])
                changes.append([np.median(a[pos]) - np.median(b[pos]),
                                100 * ((b[pos] <= 25).mean() - (a[pos] <= 25).mean())])
            ci = np.quantile(changes, [.025, .975], axis=0)
            result[label].update(blocks=len(members), block_bootstrap_repetitions=1000,
                                 median_gain_km_ci95=ci[:, 0].tolist(), acc25_gain_pp_ci95=ci[:, 1].tolist())
    return result


def train(args, dataset):
    import torch
    from models.geoclip_time.lib.temporal_training import run_prediction, run_training

    manifest = verify_dataset(dataset)
    baseline = args.baseline
    if baseline is None:
        options = sorted(RUNS.glob("t1_full_*/comparison.json"))
        if not options:
            raise FileNotFoundError("Local baseline has not finished (missing comparison.json).")
        baseline = options[-1].parent
    baseline_log = read_json(baseline / "results/metrics" / f"{BASE_NAME}_train_log.json")
    if baseline_log["validation_ids"] != manifest["validation_and_test"]["validation_ids"]:
        raise ValueError("Local baseline validation selection does not match augmented data.")
    for section, val in ORIGINAL_CONFIG.items():
        if section not in {"data_root", "run_name", "split_csv"} and baseline_log["config"][section] != val:
            raise ValueError(f"Local baseline configuration differs at {section}.")

    torch.set_num_threads(4)
    directory = args.store / "t1_runs"
    signature = _sha256(dataset / "build_manifest.json")
    reusable = []
    for receipt in sorted(directory.glob("*/training.json")):
        details = read_json(receipt)
        if details["build_signature"] == signature and _sha256(receipt.parent / "ckpt" / f"{NAME}.pt") == details["checkpoint_sha256"]:
            reusable.append((receipt.parent, details))
    if reusable:
        run, details = reusable[-1]
        print(f"Using completed training phase {run}", flush=True)
    else:
        run = directory / datetime.now(timezone.utc).strftime("aug_%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True)
        for name in DATA_FILES:
            target = run / name
            target.parent.mkdir(exist_ok=True)
            try:
                os.link(dataset / name, target)
            except OSError:
                shutil.copyfile(dataset / name, target)
        cfg = {**ORIGINAL_CONFIG, "run_name": NAME, "data_root": str(run), "split_csv": str(run / DATA_FILES[3])}
        print(f"Starting T1 from pretrained weights in {run}", flush=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        fit = run_training(cfg, device="cuda")
        torch.cuda.synchronize()
        elapsed = perf_counter() - started
        if fit["validation_ids"] != baseline_log["validation_ids"]:
            raise ValueError("Different validation queries used in actual fit.")
        details = {"build_signature": signature, "training_seconds": elapsed,
                   "checkpoint_sha256": _sha256(fit["checkpoint"]), "training_peak_mib": torch.cuda.max_memory_allocated() / 2**20}
        _atomic_json(run / "training.json", details)
        del fit
        torch.cuda.empty_cache()
    metrics_file = run / "results/metrics" / f"{NAME}_summary.json"
    output = run / "results/preds" / f"{NAME}.parquet"
    if not (run / "evaluation.json").exists():
        started = perf_counter()
        run_prediction(run / "ckpt" / f"{NAME}.pt", device="cuda", allow_partial_features=True, overwrite=True)
        torch.cuda.synchronize()
        _atomic_json(run / "evaluation.json", {"evaluation_seconds": perf_counter() - started,
                                               "predictions_sha256": _sha256(output)})
    evaluation = read_json(run / "evaluation.json")
    if _sha256(output) != evaluation["predictions_sha256"]:
        raise ValueError("Predictions changed after evaluation.")
    before = pd.read_parquet(baseline / "results/preds" / f"{BASE_NAME}.parquet")
    after = pd.read_parquet(output)
    result = paired_results(before, after, pd.read_parquet(ORIGINAL / DATA_FILES[2], columns=["id","latitude","longitude"]),
                            pd.read_csv(SPLIT, usecols=["id", "split", "block"]))
    local = read_json(baseline / "comparison.json")
    measured = {r["split"]:r for r in read_json(metrics_file)["splits"]}
    for label, section in result.items():
        if section["n"] != local["splits"][label]["n"] or not np.isclose(section["augmented"]["median_km"], measured[label]["median_km"]):
            raise ValueError(f"Mismatched comparable outputs for {label}.")
    history = read_json(run / "results/metrics" / f"{NAME}_train_log.json")
    comparison = {**details, **evaluation, "baseline": str(baseline), "dataset": str(dataset), "run": str(run),
                  "gpu":torch.cuda.get_device_name(), "epochs": len(history["history"]), "best_epoch":history["epoch"],
                  "baseline_best_epoch":baseline_log["epoch"], "baseline_training_seconds":local["training_seconds"],
                  "baseline_evaluation_seconds":local["evaluation_seconds"],
                  "augmented_examples_per_epoch":history["history"][0]["train_examples"],
                  "augmented_updates_per_epoch":history["history"][0]["optimizer_steps"],
                  "baseline_updates_per_epoch":baseline_log["history"][0]["optimizer_steps"],
                  "splits": result}
    _atomic_json(run / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("calibrate", "build", "train", "all"))
    parser.add_argument("--store", type=Path, default=VOLUME / "aug8192")
    parser.add_argument("--images", type=Path, default=VOLUME / "public-assets/images")
    parser.add_argument("--clip", type=Path, default=VOLUME / "public-assets/clip")
    parser.add_argument("--baseline", type=Path, help="Completed run_original_t1.py run directory")
    parser.add_argument("--batch-size", type=int, default=16, help="CLIP feature extraction batch size")
    parser.add_argument("--device", choices=("cuda","cpu"), default="cuda", help="Device for calibrate only")
    args = parser.parse_args()
    if args.action == "calibrate":
        args.store.mkdir(parents=True, exist_ok=True)
    if not args.store.is_dir() or args.batch_size < 1:
        parser.error("Supply an existing generator store and positive batch size.")
    args.store = args.store.resolve()
    with portalocker.Lock(str(args.store / ".augment_train.lock"), timeout=0):
        if args.action == "calibrate":
            calibrate(args)
            return
        dataset = args.store / "t1_dataset"
        if args.action in ("build", "all"):
            dataset = build(args)
        if args.action in ("train", "all"):
            train(args, dataset)


if __name__ == "__main__":
    main()
