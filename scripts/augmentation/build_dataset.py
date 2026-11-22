#!/usr/bin/env python3
"""Publish an isolated training view from accepted diffusion images.

Original metadata, split CSV, image files, CLIP arrays and the synthetic SQLite
store are read-only. Original rows retain their order and raw fp16 features;
synthetic train rows are appended. A sibling staging directory supports resume
and is renamed into the requested output root only after validation completes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_NAME = "clip_vitl14"
EXPECTED_TRANSFORMERS = "5.22.2"
BUILD_VERSION = 1


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _ids(values, label):
    series = pd.Series(values)
    if series.isna().any() or series.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(f"{label} has null or boolean IDs")
    numbers = pd.to_numeric(series, errors="raise")
    if not np.isfinite(numbers.to_numpy()).all() or (numbers != np.floor(numbers)).any():
        raise ValueError(f"{label} IDs must be finite integers")
    bounds = np.iinfo(np.int64)
    if any(int(value) < bounds.min or int(value) > bounds.max for value in numbers):
        raise ValueError(f"{label} IDs exceed int64")
    result = numbers.to_numpy(dtype=np.int64)
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{label} has duplicate IDs")
    return result


def _store_image(store, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"synthetic image_path must be relative to store: {relative!r}")
    resolved = (store / path).resolve()
    if not resolved.is_relative_to((store / "images").resolve()):
        raise ValueError("accepted synthetic images must be inside store/images")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def load_accepted_samples(store):
    """Read one SQLite snapshot; reject malformed accepted provenance."""
    store = Path(store).resolve()
    database = store / "synthetic.sqlite"
    uri = database.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(samples)")}
        required = {
            "synthetic_id", "parent_id", "parent_split", "is_synthetic", "is_diffusion",
            "source", "generator", "status", "image_path", "parent_path", "parent_sha256",
            "image_sha256", "width", "height", "model_revision", "model_sha256", "parameters_json",
        }
        if required - columns:
            raise ValueError(f"samples missing columns: {sorted(required - columns)}")
        rows = [dict(row) for row in connection.execute(
            "SELECT * FROM samples WHERE status = 'accepted' ORDER BY parent_id"
        )]
    if not rows:
        raise ValueError("store has no accepted diffusion samples")
    _ids([row["parent_id"] for row in rows], "synthetic parents")
    _ids([row["synthetic_id"] for row in rows], "synthetic samples")
    for row in rows:
        parent_id, synthetic_id = row["parent_id"], row["synthetic_id"]
        if parent_id <= 0 or synthetic_id != -parent_id or row["parent_split"] != "train":
            raise ValueError(f"invalid train-only parent/synthetic ID contract: {parent_id}")
        if (row["is_synthetic"], row["is_diffusion"], row["source"], row["generator"]) != (1, 1, "diffusion", "sdxl_img2img"):
            raise ValueError(f"accepted sample {synthetic_id} is not verified diffusion provenance")
        path = _store_image(store, row["image_path"])
        if _sha256(path) != row["image_sha256"]:
            raise ValueError(f"synthetic image sha256 mismatch: {path}")
    return rows


def _evaluation_audit(original_ids, augmented_ids, original_splits, augmented_splits, *, seed=0, limit=256):
    source = original_splits.set_index("id").loc[original_ids, "split"].to_numpy()
    target = augmented_splits.set_index("id").loc[augmented_ids, "split"].to_numpy()
    checks = {}
    for label in sorted(set(source)):
        if label == "train":
            continue
        before = original_ids[source == label]
        after = augmented_ids[target == label]
        if not np.array_equal(before, after):
            raise ValueError(f"augmentation changed original {label} feature-ID order")
        checks[str(label)] = {"count": len(before), "ids_sha256": _json_digest(before.tolist())}
    before_val = np.flatnonzero(source == "val")
    after_val = np.flatnonzero(target == "val")
    selections = []
    for positions, ids in ((before_val, original_ids), (after_val, augmented_ids)):
        if len(positions) > limit:
            positions = np.sort(np.random.default_rng(seed).choice(positions, size=limit, replace=False))
        selections.append(ids[positions].tolist())
    if selections[0] != selections[1]:
        raise ValueError("augmentation changed the deterministic validation subset")
    return {"evaluation_splits": checks, "validation_seed": seed, "validation_limit": limit,
            "validation_ids": selections[0]}


def make_augmented_tables(metadata, splits, original_ids, accepted, store, *, validation_seed=0, validation_limit=256):
    """Copy parent labels from original tables, appending synthetic train rows."""
    metadata, splits = metadata.copy(), splits.copy()
    needed = {"id", "latitude", "longitude", "date_shot_min", "date_shot_max"}
    if needed - set(metadata) or {"id", "split"} - set(splits):
        raise ValueError("original metadata/split schema is incomplete")
    metadata["id"] = _ids(metadata.id, "metadata")
    splits["id"] = _ids(splits.id, "splits")
    original_ids = _ids(original_ids, "original feature cache")
    if (metadata.id <= 0).any() or (splits.id <= 0).any() or (original_ids <= 0).any():
        raise ValueError("source dataset must contain only original positive IDs")
    if splits["split"].isna().any() or splits["split"].astype(str).str.strip().eq("").any():
        raise ValueError("source splits contain missing labels")
    cached = set(original_ids.tolist())
    if not cached <= set(metadata.id) or not cached <= set(splits.id):
        raise ValueError("original cached IDs are missing metadata or splits")
    parents = _ids([row["parent_id"] for row in accepted], "accepted parents")
    synthetic_ids = _ids([row["synthetic_id"] for row in accepted], "accepted synthetic samples")
    if not np.array_equal(synthetic_ids, -parents):
        raise ValueError("synthetic IDs must equal -parent_id")
    if not set(parents) <= cached:
        raise ValueError("every synthetic parent must already have an original cached feature")
    parent_splits = splits.set_index("id").loc[parents]
    if not parent_splits["split"].eq("train").all():
        raise ValueError("synthetic parents must belong to exact train split")
    synthetic = metadata.set_index("id", drop=False).loc[parents].reset_index(drop=True).copy()
    synthetic["id"] = synthetic_ids
    provenance = {
        "is_synthetic": [True] * len(accepted), "is_diffusion": [True] * len(accepted),
        "parent_id": pd.array(parents, dtype="Int64"),
        "augmentation_store": [str(Path(store).resolve())] * len(accepted),
    }
    for name in ("source", "generator", "image_path", "parent_sha256", "image_sha256",
                 "model_revision", "model_sha256", "parameters_json"):
        provenance["augmentation_" + name] = [row[name] for row in accepted]
    for name, values in provenance.items():
        if name in metadata:
            raise ValueError(f"source metadata already has augmentation provenance column {name}")
        metadata[name] = False if name in {"is_synthetic", "is_diffusion"} else pd.NA
        if name == "parent_id":
            metadata[name] = pd.array([pd.NA] * len(metadata), dtype="Int64")
        synthetic[name] = values
    for dimension in ("width", "height"):
        if dimension in synthetic:
            synthetic[dimension] = [int(row[dimension]) for row in accepted]
    if "image_url" in synthetic:
        synthetic["image_url"] = [(Path(store) / row["image_path"]).resolve().as_uri() for row in accepted]
    augmented_meta = pd.concat([metadata, synthetic], ignore_index=True)
    synthetic_splits = parent_splits.reset_index().copy()
    synthetic_splits["id"], synthetic_splits["split"] = synthetic_ids, "train"
    if "random_split" in synthetic_splits:
        synthetic_splits["random_split"] = "train"
    augmented_splits = pd.concat([splits, synthetic_splits[splits.columns]], ignore_index=True)
    augmented_ids = np.concatenate([original_ids, synthetic_ids])
    audit = _evaluation_audit(original_ids, augmented_ids, splits, augmented_splits,
                              seed=validation_seed, limit=validation_limit)
    audit.update({"original_feature_rows": len(original_ids), "synthetic_rows": len(accepted),
                  "total_feature_rows": len(augmented_ids)})
    return augmented_meta, augmented_splits, audit


def validate_parent_features(original, recomputed, parent_ids, *, min_cosine=0.999, max_norm_relative_error=0.02):
    """Reject changed preprocessing, normalized features or the wrong backbone."""
    original, recomputed = np.asarray(original, dtype=np.float64), np.asarray(recomputed, dtype=np.float64)
    if original.shape != recomputed.shape or original.ndim != 2 or len(original) != len(parent_ids) or not len(original):
        raise ValueError("parent CLIP comparison shapes do not match")
    if not np.isfinite(original).all() or not np.isfinite(recomputed).all():
        raise ValueError("parent CLIP comparison has nonfinite features")
    left, right = np.linalg.norm(original, axis=1), np.linalg.norm(recomputed, axis=1)
    if (left == 0).any() or (right == 0).any():
        raise ValueError("parent CLIP features have zero amplitude")
    cosine = (original * recomputed).sum(axis=1) / (left * right)
    ratios = right / left
    relative_l2 = np.linalg.norm(original - recomputed, axis=1) / left
    if (cosine < min_cosine).any() or (np.abs(ratios - 1) > max_norm_relative_error).any():
        raise ValueError(f"parent CLIP calibration failed: min cosine={cosine.min():.6f}, norm ratio range={ratios.min():.6f}..{ratios.max():.6f}")
    return {"parent_ids": [int(value) for value in parent_ids], "min_cosine": float(cosine.min()),
            "mean_cosine": float(cosine.mean()), "min_norm_ratio": float(ratios.min()),
            "max_norm_ratio": float(ratios.max()), "max_relative_l2": float(relative_l2.max()),
            "minimum_required_cosine": min_cosine, "maximum_norm_relative_error": max_norm_relative_error}


class FrozenClipEncoder:
    """The original GeoCLIP preprocessing and raw CLIP pre-MLP feature helper."""

    def __init__(self, clip_model, *, cache_dir=None, device="cuda", expected_transformers=EXPECTED_TRANSFORMERS):
        installed = importlib.metadata.version("transformers")
        if installed != expected_transformers:
            raise ValueError(f"original cache uses transformers {expected_transformers}; installed {installed}")
        import torch
        from transformers import AutoProcessor, CLIPModel
        from geoclip.model.image_encoder import ImageEncoder
        from lib.features import encode_clip_pre_mlp

        self.device = torch.device(device)
        self.CLIP = CLIPModel.from_pretrained(
            str(clip_model), cache_dir=cache_dir, local_files_only=True, dtype=torch.float32,
        ).to(self.device).eval()
        self.CLIP.requires_grad_(False)
        self.image_processor = AutoProcessor.from_pretrained(
            str(clip_model), cache_dir=cache_dir, local_files_only=True,
        )
        self._preprocess, self._extract = ImageEncoder.preprocess_image, encode_clip_pre_mlp
        self.info = {"model": str(clip_model), "cache_dir": str(cache_dir) if cache_dir else None,
                     "transformers": installed, "torch": importlib.metadata.version("torch"),
                     "geoclip": importlib.metadata.version("geoclip"), "dtype": "raw float16",
                     "preprocessing": "geoclip.model.image_encoder.ImageEncoder.preprocess_image",
                     "feature_helper": "lib.features.encode_clip_pre_mlp", "local_files_only": True}

    def encode(self, paths):
        import torch
        from PIL import Image

        images = []
        try:
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            with torch.inference_mode():
                pixels = self._preprocess(self, images)
                if pixels.ndim == 3:
                    pixels = pixels.unsqueeze(0)
                raw = self._extract(self, pixels.to(self.device))
                if raw.ndim != 2 or raw.shape != (len(paths), 768) or not torch.isfinite(raw).all():
                    raise ValueError("CLIP must emit finite raw pre-MLP features with shape (N, 768)")
                return raw.float().cpu().numpy()
        finally:
            for image in images:
                image.close()


def _check_features(features, rows):
    if features.shape != (rows, 768) or features.dtype != np.float16:
        raise ValueError(f"original cache must be raw fp16 (N,768); found {features.shape}, {features.dtype}")
    for start in range(0, rows, 8192):
        if not np.isfinite(features[start:start + 8192]).all():
            raise ValueError(f"feature cache has nonfinite values near row {start}")


def inspect_inputs(source_data_root, source_split_csv, store, *, images_root=None, validation_seed=0, validation_limit=256):
    """Validate a fixed accepted snapshot without constructing a CLIP model."""
    source, store = Path(source_data_root).resolve(), Path(store).resolve()
    split_path = Path(source_split_csv).resolve()
    accepted = load_accepted_samples(store)
    paths = {"metadata": source / "meta/smapshot.parquet", "splits": split_path,
             "features": source / f"features/{FEATURE_NAME}.npy", "ids": source / f"features/{FEATURE_NAME}_ids.npy"}
    metadata, splits = pd.read_parquet(paths["metadata"]), pd.read_csv(split_path)
    original_ids = _ids(np.load(paths["ids"], allow_pickle=False), "feature cache")
    features = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
    _check_features(features, len(original_ids))
    meta, split, audit = make_augmented_tables(metadata, splits, original_ids, accepted, store,
        validation_seed=validation_seed, validation_limit=validation_limit)
    images = Path(images_root).resolve() if images_root else source / "images"
    for row in accepted:
        parent_id = int(row["parent_id"])
        path = images / f"{parent_id // 1000:04d}" / f"{parent_id}.jpg"
        if not path.is_file() or _sha256(path) != row["parent_sha256"]:
            raise ValueError(f"parent image missing or sha256 differs from generation input: {path}")
    if not (source / "geo").is_dir():
        raise FileNotFoundError(source / "geo")
    fingerprints = {name: {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                            "sha256": _sha256(path)} for name, path in paths.items()}
    return {"source": source, "store": store, "images": images, "accepted": accepted,
            "features": features, "ids": original_ids, "metadata": meta, "splits": split,
            "audit": audit, "source_files": fingerprints}


@contextmanager
def _build_lock(output):
    import fcntl

    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.build.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another builder holds the output lock: {output}") from error
        yield


def _publish(stage, output):
    manifest_path = stage / "build_manifest.json"
    if not manifest_path.is_file() or json.loads(manifest_path.read_text()).get("status") != "complete":
        raise ValueError("refusing to publish an incomplete dataset")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    os.rename(stage, output)


def build_dataset(source_data_root, source_split_csv, store, output_data_root, *,
                  images_root=None, clip_model="openai/clip-vit-large-patch14", clip_cache_dir=None,
                  device="cuda", batch_size=32, parent_check_count=8, min_parent_cosine=0.999,
                  max_parent_norm_relative_error=0.02, expected_transformers=EXPECTED_TRANSFORMERS,
                  validation_seed=0, validation_limit=256, encoder=None):
    """Build or resume an atomic dataset; identical completed inputs are reused."""
    if batch_size < 1 or parent_check_count < 1 or validation_limit < 1:
        raise ValueError("batch size, parent check count and validation limit must be positive")
    if not 0 < min_parent_cosine <= 1 or not 0 <= max_parent_norm_relative_error < 1:
        raise ValueError("invalid parent feature comparison thresholds")
    output = Path(output_data_root).resolve()
    for original in (Path(source_data_root).resolve(), Path(store).resolve()):
        if output == original or output.is_relative_to(original) or original.is_relative_to(output):
            raise ValueError("output must be separate from original data and the synthetic store")
    inputs = inspect_inputs(source_data_root, source_split_csv, store, images_root=images_root,
                            validation_seed=validation_seed, validation_limit=validation_limit)
    configuration = {"version": BUILD_VERSION, "clip_model": str(clip_model),
        "clip_cache_dir": str(clip_cache_dir) if clip_cache_dir else None,
        "expected_transformers": expected_transformers, "parent_check_count": parent_check_count,
        "min_parent_cosine": min_parent_cosine, "max_parent_norm_relative_error": max_parent_norm_relative_error,
        "validation_seed": validation_seed, "validation_limit": validation_limit,
        "encoder_override": getattr(encoder, "info", None)}
    signature = _json_digest({"configuration": configuration, "source_files": inputs["source_files"],
                             "accepted": inputs["accepted"], "store": str(inputs["store"])})
    with _build_lock(output):
        if output.exists():
            path = output / "build_manifest.json"
            if not path.is_file():
                raise FileExistsError(f"output exists without a completed build manifest: {output}")
            existing = json.loads(path.read_text())
            if existing.get("status") != "complete" or existing.get("input_signature") != signature:
                raise FileExistsError("output belongs to different inputs; choose a new output data root")
            for relative, digest in existing["artifacts"].items():
                if _sha256(output / relative) != digest:
                    raise ValueError(f"completed output artifact changed: {relative}")
            return existing
        stage = output.parent / f".{output.name}.building-{signature[:16]}"
        for directory in (stage / "meta", stage / "features", stage / "splits"):
            directory.mkdir(parents=True, exist_ok=True)
        if encoder is None:
            encoder = FrozenClipEncoder(clip_model, cache_dir=clip_cache_dir, device=device,
                                        expected_transformers=expected_transformers)
        selected = inputs["accepted"][:parent_check_count]
        parent_ids = [int(row["parent_id"]) for row in selected]
        parent_paths = [inputs["images"] / f"{parent // 1000:04d}/{parent}.jpg" for parent in parent_ids]
        recomputed = np.concatenate([encoder.encode(parent_paths[start:start + batch_size])
                                     for start in range(0, len(parent_paths), batch_size)])
        positions = pd.Index(inputs["ids"]).get_indexer(parent_ids)
        calibration = validate_parent_features(inputs["features"][positions], recomputed, parent_ids,
            min_cosine=min_parent_cosine, max_norm_relative_error=max_parent_norm_relative_error)
        print(json.dumps({"parent_calibration": calibration}), flush=True)

        progress_path = stage / "progress.json"
        merged_path = stage / f"features/{FEATURE_NAME}.npy"
        original_count, synthetic_count = len(inputs["ids"]), len(inputs["accepted"])
        total = original_count + synthetic_count
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress.get("input_signature") != signature:
                raise ValueError("staging signature mismatch")
            merged = np.lib.format.open_memmap(merged_path, mode="r+")
            if merged.shape != (total, 768) or merged.dtype != np.float16:
                raise ValueError("staging feature array shape/dtype mismatch")
            completed = int(progress["synthetic_completed"])
            if not 0 <= completed <= synthetic_count:
                raise ValueError("invalid staging progress count")
        else:
            merged = np.lib.format.open_memmap(merged_path, mode="w+", dtype=np.float16, shape=(total, 768))
            for start in range(0, original_count, 8192):
                merged[start:min(start + 8192, original_count)] = inputs["features"][start:start + 8192]
            merged.flush()
            completed = 0
            _atomic_json(progress_path, {"input_signature": signature, "synthetic_completed": 0})
        for start in range(completed, synthetic_count, batch_size):
            rows = inputs["accepted"][start:start + batch_size]
            paths = [_store_image(inputs["store"], row["image_path"]) for row in rows]
            raw = np.asarray(encoder.encode(paths))
            for row, path in zip(rows, paths):
                if _sha256(path) != row["image_sha256"]:
                    raise ValueError(f"synthetic image sha256 changed during encoding: {path}")
            if raw.shape != (len(rows), 768) or not np.isfinite(raw).all():
                raise ValueError("synthetic CLIP features must be finite (N,768)")
            half = raw.astype(np.float16)
            if not np.isfinite(half).all():
                raise ValueError("synthetic CLIP features overflow fp16")
            merged[original_count + start:original_count + start + len(rows)] = half
            merged.flush()
            _atomic_json(progress_path, {"input_signature": signature, "synthetic_completed": start + len(rows)})
            print(f"cached synthetic features {start + len(rows)}/{synthetic_count}", flush=True)
        _check_features(merged, total)
        for start in range(0, original_count, 8192):
            stop = min(start + 8192, original_count)
            if not np.array_equal(merged[start:stop], inputs["features"][start:stop]):
                raise ValueError("original cached feature prefix was changed")
        merged.flush()
        del merged
        merged_ids = np.concatenate([inputs["ids"], np.array([row["synthetic_id"] for row in inputs["accepted"]], dtype=np.int64)])
        np.save(stage / f"features/{FEATURE_NAME}_ids.npy", merged_ids)
        inputs["metadata"].to_parquet(stage / "meta/smapshot.parquet", index=False)
        inputs["splits"].to_csv(stage / "splits/main.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        geometry = stage / "geo"
        if not geometry.is_symlink():
            geometry.symlink_to(inputs["source"] / "geo", target_is_directory=True)
        elif geometry.resolve() != (inputs["source"] / "geo").resolve():
            raise ValueError("staging geo symlink points at the wrong source")
        _atomic_json(stage / f"features/{FEATURE_NAME}_meta.json", {
            "n_images": total, "feat_dim": 768, "dtype": "float16", "normalized": False,
            "encoder": encoder.info, "parent_calibration": calibration,
            "original_feature_rows": original_count, "synthetic_rows": synthetic_count,
        })
        for item in inputs["source_files"].values():
            stat = Path(item["path"]).stat()
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError(f"original source changed during build: {item['path']}")
        for row in inputs["accepted"]:
            image = _store_image(inputs["store"], row["image_path"])
            if _sha256(image) != row["image_sha256"]:
                raise ValueError(f"synthetic image sha256 changed during build: {image}")
        relative_artifacts = ["meta/smapshot.parquet", "splits/main.csv.gz",
            f"features/{FEATURE_NAME}.npy", f"features/{FEATURE_NAME}_ids.npy", f"features/{FEATURE_NAME}_meta.json"]
        manifest = {"status": "complete", "input_signature": signature, "configuration": configuration,
            "source_files": inputs["source_files"], "store": str(inputs["store"]),
            "accepted_snapshot_sha256": _json_digest(inputs["accepted"]),
            "synthetic_ids": [row["synthetic_id"] for row in inputs["accepted"]],
            "parent_calibration": calibration, "encoder": encoder.info, **inputs["audit"],
            "artifacts": {name: _sha256(stage / name) for name in relative_artifacts}}
        _atomic_json(stage / "build_manifest.json", manifest)
        _publish(stage, output)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root", required=True, type=Path)
    parser.add_argument("--source-split-csv", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path, help="independent store containing synthetic.sqlite and images/")
    parser.add_argument("--output-data-root", required=True, type=Path)
    parser.add_argument("--images-root", type=Path, help="original image directory (default: source-data-root/images)")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14", help="existing local snapshot or already-cached HF model; never downloads")
    parser.add_argument("--clip-cache-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--parent-check-count", type=int, default=8)
    parser.add_argument("--min-parent-cosine", type=float, default=0.999)
    parser.add_argument("--max-parent-norm-relative-error", type=float, default=0.02)
    parser.add_argument("--expected-transformers", default=EXPECTED_TRANSFORMERS)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=256)
    parser.add_argument("--check-only", action="store_true", help="validate data/accepted hashes without loading CLIP or publishing")
    args = vars(parser.parse_args())
    check_only = args.pop("check_only")
    if check_only:
        inputs = inspect_inputs(args["source_data_root"], args["source_split_csv"], args["store"],
            images_root=args["images_root"], validation_seed=args["validation_seed"], validation_limit=args["validation_limit"])
        print(json.dumps({"status": "inputs_valid", **inputs["audit"]}, indent=2))
    else:
        print(json.dumps(build_dataset(**args), indent=2))


if __name__ == "__main__":
    main()
