"""Compare archived SDXL variant CLIP features with their original parent photos."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

for setting in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[setting] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "report/experiments/2026-11-20-diffusion-augmentation/augmented/build_manifest.json"
ORIGINAL = ROOT / "data/t1_original/features"
RESULTS = ROOT / "data/local_runs/clip_variants"


def merged_digest(original, tail, relative_path, manifest):
    """Hash the exact .npy file open_memmap/np.save would write without copying it."""
    if original.dtype != tail.dtype or not original.flags.c_contiguous:
        raise ValueError(f"Unexpected cache dtype or array order: {relative_path}")
    shape = (len(original) + len(tail), *original.shape[1:])
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": original.dtype.str, "fortran_order": False, "shape": shape}
    )
    digest = hashlib.sha256(header.getvalue())
    for start in range(0, len(original), 4096):
        digest.update(memoryview(original[start:start + 4096]).cast("B"))
    digest.update(memoryview(tail).cast("B"))
    value = digest.hexdigest()
    if value != manifest["artifacts"][relative_path]:
        raise ValueError(f"Reconstructed SHA-256 differs from archive: {relative_path}: {value}")
    return value


def stats(values):
    levels = [0, 5, 25, 50, 75, 95, 100]
    quantiles = np.percentile(values, levels)
    return {"mean": float(np.mean(values)), **{f"p{k}": float(v) for k, v in zip(levels, quantiles)}}


def compare(tail):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Augmented build is not complete.")
    synthetic_ids = np.array(manifest["synthetic_ids"], dtype="<i8")
    count = len(synthetic_ids)
    expected_bytes = count * 768 * np.dtype("<f2").itemsize
    if tail.stat().st_size != expected_bytes or count != manifest["synthetic_rows"]:
        raise ValueError("Variant row count or byte length differs from archive.")
    if np.any(synthetic_ids >= 0) or np.unique(synthetic_ids).size != count:
        raise ValueError("Synthetic IDs must be unique and negative.")
    variants = np.fromfile(tail, dtype="<f2").reshape(count, 768)
    original = np.load(ORIGINAL / "clip_vitl14.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(ORIGINAL / "clip_vitl14_ids.npy", allow_pickle=False)
    if original.shape != (manifest["original_feature_rows"], 768) or ids.shape != (len(original),):
        raise ValueError("Original cache shape does not match archived build.")
    hashes = {
        "features/clip_vitl14_ids.npy": merged_digest(ids, synthetic_ids, "features/clip_vitl14_ids.npy", manifest),
        "features/clip_vitl14.npy": merged_digest(original, variants, "features/clip_vitl14.npy", manifest),
    }
    parent_ids = -synthetic_ids
    order = np.argsort(ids)
    sorted_ids = ids[order]
    if np.any(np.diff(sorted_ids) <= 0) or sorted_ids[0] <= 0:
        raise ValueError("Original IDs must be unique and positive.")
    locations = np.searchsorted(sorted_ids, parent_ids)
    if np.any(locations >= len(ids)) or not np.array_equal(sorted_ids[locations], parent_ids):
        raise ValueError("A parent feature is missing.")
    parents = np.asarray(original[order[locations]], dtype=np.float64)
    images = np.asarray(variants, dtype=np.float64)
    parent_norms = np.linalg.norm(parents, axis=1)
    image_norms = np.linalg.norm(images, axis=1)
    if not np.all(np.isfinite(image_norms + parent_norms)) or min(parent_norms.min(), image_norms.min()) <= 0:
        raise ValueError("Non-finite or zero feature norm.")
    similarities = (images / image_norms[:, None]) @ (parents / parent_norms[:, None]).T
    paired = similarities.diagonal().copy()
    other = similarities[~np.eye(count, dtype=bool)].reshape(count, count - 1)
    best_other = other.max(axis=1)
    ranks = 1 + (other > paired[:, None]).sum(axis=1)
    return {
        "parent_variant_pairs": count,
        "embedding": "Raw CLIP ViT-L/14 cached feature before the GeoCLIP image MLP (768 dimensions).",
        "manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "verified_reconstructed_augmented_npy_sha256": hashes,
        "raw_tail_sha256": hashlib.sha256(memoryview(variants).cast("B")).hexdigest(),
        "n_identical_cached_vectors": int(np.all(parents == images, axis=1).sum()),
        "paired_cosine": stats(paired),
        "paired_angle_degrees": stats(np.degrees(np.arccos(np.clip(paired, -1, 1)))),
        "relative_l2_difference_to_parent_norm": stats(np.linalg.norm(images - parents, axis=1) / parent_norms),
        "variant_norm_divided_by_parent_norm": stats(image_norms / parent_norms),
        "n_other_pairs": int(other.size),
        "nonmatched_cosine_within_512_parents": stats(other),
        "best_other_cosine_by_variant": stats(best_other),
        "parent_cosine_minus_best_other": stats(paired - best_other),
        "n_parent_unique_top1_among_512_parents": int((paired > best_other).sum()),
        "n_parent_tied_for_top1_among_512_parents": int((paired == best_other).sum()),
        "parent_rank_among_512_parents_competition_ranking": stats(ranks),
        "min_cosine_parent_id": int(parent_ids[np.argmin(paired)]),
        "max_cosine_parent_id": int(parent_ids[np.argmax(paired)]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail", type=Path, default=RESULTS / "variant_rows.bin")
    parser.add_argument("--output", type=Path, default=RESULTS / "metrics.json")
    args = parser.parse_args()
    result = compare(args.tail)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output, flush=True)
