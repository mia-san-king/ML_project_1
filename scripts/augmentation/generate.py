#!/usr/bin/env python3
"""Generate one locally loaded SDXL img2img variant per cached train photo.

Original inputs are read-only. PNGs and a resumable provenance/quality database
live under --store. Only samples.status='accepted' may enter an augmented set.
This structural prescreen is not a guarantee that geographic content survived;
inspect parent/variant pairs before starting a larger experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import sqlite3


PROMPT = "The same historical documentary photograph, same scene, same viewpoint, same framing, same buildings and landscape. Subtle photographic variation in exposure and texture, preserve the original composition."
NEGATIVE_PROMPT = "different viewpoint, reframing, zoom, crop, new buildings, removed buildings, moved landmarks, changed geometry, illustration, painting, text, watermark"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_image(image, *, max_side=768):
    """Uniformly resize full content, then add only temporary edge padding."""
    import numpy as np
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    original = np.asarray(image, dtype=np.int16)
    chroma = original.max(axis=2) - original.min(axis=2)
    grayscale = bool(chroma.mean() <= 2.0 and np.percentile(chroma, 95) <= 5.0)
    scale = max_side / max(image.size)
    content_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    content = image.resize(content_size, Image.Resampling.LANCZOS)
    canvas_size = tuple(max(512, math.ceil(side / 64) * 64) for side in content_size)
    left, top = (canvas_size[0] - content_size[0]) // 2, (canvas_size[1] - content_size[1]) // 2
    right, bottom = canvas_size[0] - content_size[0] - left, canvas_size[1] - content_size[1] - top
    pixels = np.pad(np.asarray(content), ((top, bottom), (left, right), (0, 0)), mode="edge")
    geometry = {"parent_size": list(image.size), "content_size": list(content_size),
                "canvas_size": list(canvas_size), "padding_ltrb": [left, top, right, bottom],
                "resize": "uniform LANCZOS, integer pixel rounding; no original content cropped",
                "alignment": "edge padding to multiples of 64, minimum canvas side 512"}
    return Image.fromarray(pixels), geometry, grayscale


def restore_image(generated, geometry, *, grayscale):
    """Remove only padding added above; never resize a wrongly shaped result."""
    from PIL import Image

    if not isinstance(generated, Image.Image) or list(generated.size) != geometry["canvas_size"]:
        raise ValueError("pipeline output must have exactly the padded input size")
    left, top, _, _ = geometry["padding_ltrb"]
    width, height = geometry["content_size"]
    result = generated.crop((left, top, left + width, top + height))
    return result.convert("L" if grayscale else "RGB")


def quality_check(parent, generated, *, min_structure=0.4):
    """Lenient aligned-image prescreen, including explicit copy rejection."""
    import numpy as np
    from PIL import Image

    if parent.size != generated.size:
        return {"passed": False, "reasons": ["size_mismatch"]}
    a, b = (np.asarray(image.convert("L"), dtype=np.float64) / 255 for image in (parent, generated))
    mae, output_std = float(np.abs(a - b).mean()), float(b.std())
    # A global SSIM-style structural statistic is intentionally only a prescreen.
    aa, bb = (np.asarray(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float64) / 255
              for image in (parent, generated))
    mean_a, mean_b = aa.mean(), bb.mean()
    covariance = ((aa - mean_a) * (bb - mean_b)).mean()
    structure = float(((2 * mean_a * mean_b + .01 ** 2) * (2 * covariance + .03 ** 2)) /
                      ((mean_a ** 2 + mean_b ** 2 + .01 ** 2) * (aa.var() + bb.var() + .03 ** 2)))
    reasons = []
    if mae < .001:
        reasons.append("difference_too_small")
    if mae > .35:
        reasons.append("excessive_pixel_change")
    if structure < min_structure:
        reasons.append("structure_changed")
    if output_std < .01:
        reasons.append("near_blank_output")
    return {"passed": not reasons, "reasons": reasons, "pixel_mae": mae, "global_structure": structure,
            "output_luma_std": output_std,
            "thresholds": {"min_pixel_mae": .001, "max_pixel_mae": .35, "min_structure": min_structure, "min_output_std": .01},
            "scope": "automatic structural prescreen; manual geographic-content review still recommended"}


def _positive_ids(values, label):
    import numpy as np

    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or (values <= 0).any():
        raise ValueError(f"{label} IDs must all be positive integers")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{label} IDs must be unique")
    return values.astype(np.int64, copy=False)


def _selection(args):
    import numpy as np
    import pandas as pd

    metadata_path = args.data_root / "meta/smapshot.parquet"
    feature_ids_path = args.feature_ids or args.data_root / "features/clip_vitl14_ids.npy"
    metadata, splits = pd.read_parquet(metadata_path), pd.read_csv(args.split_csv)
    required = {"id", "latitude", "longitude", "date_shot_min", "date_shot_max"}
    if not required.issubset(metadata.columns) or not {"id", "split"}.issubset(splits.columns):
        raise ValueError("metadata/splits are missing required ID, GPS, date or split columns")
    metadata_ids = _positive_ids(metadata.id, "metadata")
    split_ids = _positive_ids(splits.id, "split")
    cache_ids = _positive_ids(np.load(feature_ids_path, allow_pickle=False), "feature cache")
    if not np.isin(cache_ids, metadata_ids).all() or not np.isin(cache_ids, split_ids).all():
        raise ValueError("every original feature ID must exist in metadata and the original split")
    train_ids = splits.loc[splits.split.eq("train"), "id"].to_numpy(dtype=np.int64)
    eligible = np.intersect1d(cache_ids, train_ids)
    if args.count > len(eligible):
        raise ValueError(f"count={args.count} exceeds eligible cached train photos={len(eligible)}")
    selected = np.random.default_rng(args.seed).permutation(eligible)[:args.count]
    frame = metadata.set_index("id").loc[selected].copy()
    gps = frame[["latitude", "longitude"]].to_numpy(dtype=float)
    if not np.isfinite(gps).all() or (np.abs(gps[:, 0]) > 90).any() or (np.abs(gps[:, 1]) > 180).any():
        raise ValueError("selected parent GPS coordinates are invalid")
    frame["block"] = splits.set_index("id").reindex(selected)["block"].to_numpy() if "block" in splits else None
    if "view_type" not in frame:
        frame["view_type"] = None
    hashes = {"metadata": _sha256(metadata_path), "split": _sha256(args.split_csv), "feature_ids": _sha256(feature_ids_path)}
    return frame, len(eligible), hashes


def _model_manifest(model_dir):
    if not (model_dir / "model_index.json").is_file():
        raise ValueError("model-dir must be a downloaded Diffusers directory containing model_index.json")
    files = sorted(path for path in model_dir.rglob("*") if path.is_file()
                   and not any(part.startswith(".") for part in path.relative_to(model_dir).parts))
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("model-dir must contain locally downloaded safetensors weights")
    manifest = []
    for path in files:
        print(f"hash local model file: {path.relative_to(model_dir)} ({path.stat().st_size} bytes)", flush=True)
        manifest.append({"path": str(path.relative_to(model_dir)), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return manifest, hashlib.sha256(_json(manifest).encode()).hexdigest()


def load_generator(args):
    """Load only the supplied local SDXL directory; never download weights."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    kwargs = {"local_files_only": True, "use_safetensors": True,
              "torch_dtype": torch.float16 if args.device == "cuda" else torch.float32, "add_watermarker": False}
    if args.variant:
        kwargs["variant"] = args.variant
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(str(args.model_dir), **kwargs).to(args.device)
    pipeline.set_progress_bar_config(disable=True)

    def render(image, *, seed, prompt, negative_prompt):
        with torch.inference_mode():
            return pipeline(image=image, prompt=prompt, negative_prompt=negative_prompt,
                            strength=args.strength, num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                            num_images_per_prompt=1, generator=torch.Generator(device="cpu").manual_seed(seed),
                            original_size=(image.height, image.width), target_size=(image.height, image.width),
                            crops_coords_top_left=(0, 0), output_type="pil").images[0]
    return render


def _initialize(conn, config, manifest, *, resume, count):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS run_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS samples (
          synthetic_id INTEGER PRIMARY KEY CHECK(synthetic_id<0), parent_id INTEGER NOT NULL UNIQUE CHECK(parent_id>0),
          selection_rank INTEGER NOT NULL UNIQUE, parent_split TEXT NOT NULL CHECK(parent_split='train'),
          is_synthetic INTEGER NOT NULL CHECK(is_synthetic=1), is_diffusion INTEGER NOT NULL CHECK(is_diffusion=1),
          source TEXT NOT NULL, generator TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','accepted','rejected','error')),
          image_path TEXT NOT NULL, latitude REAL NOT NULL, longitude REAL NOT NULL, date_shot_min TEXT, date_shot_max TEXT,
          view_type TEXT, block TEXT, seed INTEGER NOT NULL, parent_path TEXT NOT NULL, parent_sha256 TEXT,
          image_sha256 TEXT, width INTEGER, height INTEGER, parent_width INTEGER, parent_height INTEGER,
          is_grayscale INTEGER, quality_json TEXT, model_revision TEXT NOT NULL, model_sha256 TEXT NOT NULL,
          parameters_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, error TEXT,
          manual_review TEXT NOT NULL DEFAULT 'not_reviewed', CHECK(synthetic_id=-parent_id));
    """)
    previous = conn.execute("SELECT value FROM run_metadata WHERE key='config'").fetchone()
    if previous and (not resume or previous[0] != _json(config)):
        raise ValueError("existing store configuration/weights/input hashes differ, or --resume was not supplied")
    old_count = conn.execute("SELECT COALESCE(MAX(selection_rank)+1,0) FROM samples").fetchone()[0]
    if count < old_count:
        raise ValueError("count cannot shrink an existing store; extend the same selection prefix")
    conn.executemany("INSERT OR REPLACE INTO run_metadata(key,value) VALUES (?,?)",
                     [("config", _json(config)), ("model_manifest", _json(manifest)), ("planned_count", str(count)), ("complete", "false")])
    conn.commit()


def _text(value):
    import pandas as pd

    if pd.isna(value):
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def run_generation(args, *, generator_factory=load_generator):
    import numpy as np
    from PIL import Image

    for key in ("data_root", "split_csv", "model_dir", "store"):
        setattr(args, key, Path(getattr(args, key)).expanduser().resolve())
    args.images_root = Path(args.images_root or args.data_root / "images").expanduser().resolve()
    if args.feature_ids:
        args.feature_ids = Path(args.feature_ids).expanduser().resolve()
    if args.count < 1 or args.seed < 0 or not 0 < args.strength <= .35 or args.steps < 1 or int(args.steps * args.strength) < 1:
        raise ValueError("require count>=1, seed>=0, 0<strength<=0.35 and at least one effective denoising step")
    if not math.isfinite(args.guidance_scale) or not 0 <= args.guidance_scale <= 10 or not 512 <= args.max_side <= 1536:
        raise ValueError("require guidance-scale in [0,10] and max-side in [512,1536]")
    if not -1 <= args.min_structure <= 1 or not args.model_revision.strip():
        raise ValueError("require min-structure in [-1,1] and an explicit model revision")
    parents, eligible_count, input_hashes = _selection(args)
    print(f"eligible cached train photos={eligible_count}; selected={len(parents)}; selection_seed={args.seed}", flush=True)
    if args.check_data:
        missing = [int(identifier) for identifier in parents.index
                   if not (args.images_root / f"{int(identifier)//1000:04d}/{int(identifier)}.jpg").is_file()]
        if not (args.model_dir / "model_index.json").is_file():
            raise ValueError("model-dir lacks model_index.json")
        return {"eligible_train": eligible_count, "selected": len(parents), "selected_ids": [int(value) for value in parents.index],
                "missing_parent_images": missing, "input_sha256": input_hashes, "weights_loaded": False}
    manifest, model_sha = _model_manifest(args.model_dir)
    runtime = {}
    for package in ("diffusers", "transformers", "torch", "numpy", "Pillow"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = None
    parameters = {key: getattr(args, key) for key in ("seed", "strength", "steps", "guidance_scale", "max_side", "min_structure", "prompt", "negative_prompt", "device", "variant")}
    config = {"schema_version": 1, "source_sha256": _sha256(__file__), "input_sha256": input_hashes,
              "data_root": str(args.data_root), "images_root": str(args.images_root), "split_csv": str(args.split_csv),
              "model_dir": str(args.model_dir), "model_revision": args.model_revision, "model_sha256": model_sha,
              "parameters": parameters, "runtime": runtime,
              "selection": "seeded permutation of sorted original cached train IDs; extensible prefix; one variant per parent"}
    database = args.store / "synthetic.sqlite"
    if database.exists() and not args.resume:
        raise FileExistsError("store already exists; use --resume for the identical configuration")
    if not database.exists() and args.store.exists() and any(args.store.iterdir()):
        raise FileExistsError("new store must be empty")
    args.store.mkdir(parents=True, exist_ok=True)
    with (args.store / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another generator holds this store lock") from exc
        conn = sqlite3.connect(database, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            _initialize(conn, config, manifest, resume=args.resume, count=args.count)
            (args.store / "images").mkdir(exist_ok=True)
            _atomic_json(args.store / "generation_manifest.json", {**config, "model_files": manifest})
            renderer = None
            for rank, (parent_id, parent) in enumerate(parents.iterrows()):
                parent_id = int(parent_id)
                parent_path = args.images_root / f"{parent_id//1000:04d}/{parent_id}.jpg"
                image_path = f"images/{-parent_id}.png"
                existing = conn.execute("SELECT * FROM samples WHERE parent_id=?", (parent_id,)).fetchone()
                if existing and (existing["selection_rank"] != rank or existing["synthetic_id"] != -parent_id):
                    raise ValueError("stored parent IDs no longer match the deterministic selection prefix")
                if existing and existing["status"] in ("accepted", "rejected"):
                    saved = (args.store / existing["image_path"]).resolve()
                    if not saved.is_relative_to(args.store) or not saved.is_file() or _sha256(saved) != existing["image_sha256"]:
                        raise ValueError(f"saved synthetic image is missing or changed: {parent_id}")
                    if not parent_path.is_file() or _sha256(parent_path) != existing["parent_sha256"]:
                        raise ValueError(f"original parent image is missing or changed: {parent_id}")
                    continue
                seed = int(np.random.SeedSequence([args.seed, parent_id]).generate_state(1)[0])
                sample = {"synthetic_id": -parent_id, "parent_id": parent_id, "selection_rank": rank,
                          "parent_split": "train", "is_synthetic": 1, "is_diffusion": 1, "source": "diffusion",
                          "generator": "sdxl_img2img", "status": "pending", "image_path": image_path,
                          "latitude": float(parent.latitude), "longitude": float(parent.longitude),
                          "date_shot_min": _text(parent.date_shot_min), "date_shot_max": _text(parent.date_shot_max),
                          "view_type": _text(parent.view_type), "block": _text(parent.block), "seed": seed,
                          "parent_path": str(parent_path), "model_revision": args.model_revision, "model_sha256": model_sha,
                          "parameters_json": _json(parameters), "created_at": existing["created_at"] if existing else _now(), "updated_at": _now()}
                columns = ",".join(sample)
                conn.execute(f"INSERT OR REPLACE INTO samples ({columns}) VALUES ({','.join('?' for _ in sample)})", tuple(sample.values()))
                conn.commit()
                loading_model = False
                try:
                    parent_sha = _sha256(parent_path)
                    with Image.open(parent_path) as original:
                        canvas, geometry, grayscale = prepare_image(original, max_side=args.max_side)
                    if renderer is None:
                        print("loading local SDXL img2img weights (network disabled)", flush=True)
                        loading_model = True
                        renderer = generator_factory(args)
                        loading_model = False
                    prompt = args.prompt + (" Preserve black-and-white grayscale photography." if grayscale else " Preserve the original colors.")
                    generated = renderer(canvas, seed=seed, prompt=prompt, negative_prompt=args.negative_prompt)
                    output = restore_image(generated, geometry, grayscale=grayscale)
                    reference = restore_image(canvas, geometry, grayscale=grayscale)
                    quality = quality_check(reference, output, min_structure=args.min_structure)
                    temporary = (args.store / image_path).with_suffix(".png.part")
                    output.save(temporary, format="PNG")
                    temporary.replace(args.store / image_path)
                    sample_parameters = {**parameters, "prompt": prompt, "geometry": geometry, "runtime": runtime}
                    conn.execute("""UPDATE samples SET status=?,parent_sha256=?,image_sha256=?,width=?,height=?,
                                 parent_width=?,parent_height=?,is_grayscale=?,quality_json=?,parameters_json=?,updated_at=?,error=NULL
                                 WHERE parent_id=?""",
                                 ("accepted" if quality["passed"] else "rejected", parent_sha, _sha256(args.store / image_path),
                                  output.width, output.height, *geometry["parent_size"], int(grayscale), _json(quality),
                                  _json(sample_parameters), _now(), parent_id))
                    conn.commit()
                    print(f"sample={rank+1}/{len(parents)} parent={parent_id} synthetic={-parent_id} status={'accepted' if quality['passed'] else 'rejected'} quality={_json(quality)}", flush=True)
                except Exception as exc:
                    conn.execute("UPDATE samples SET status='error',error=?,updated_at=? WHERE parent_id=?", (repr(exc), _now(), parent_id))
                    conn.commit()
                    print(f"sample={rank+1}/{len(parents)} parent={parent_id} status=error error={exc!r}", flush=True)
                    if loading_model or isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                        raise
            counts = {row["status"]: row["n"] for row in conn.execute("SELECT status,COUNT(*) AS n FROM samples GROUP BY status")}
            summary = {"database": str(database), "selected": len(parents), "eligible_train": eligible_count,
                       **{name: int(counts.get(name, 0)) for name in ("accepted", "rejected", "error", "pending")},
                       "model_sha256": model_sha, "model_revision": args.model_revision,
                       "selection_ids_sha256": hashlib.sha256(np.asarray(parents.index, dtype="<i8").tobytes()).hexdigest(),
                       "training_filter": "status='accepted' AND is_synthetic=1 AND is_diffusion=1", "updated_at": _now()}
            conn.execute("INSERT OR REPLACE INTO run_metadata(key,value) VALUES('complete',?)", (_json(not summary["error"]),))
            conn.commit()
            _atomic_json(args.store / "generation_summary.json", summary)
            return summary
        finally:
            conn.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("data-root", "split-csv", "model-dir", "store"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--images-root", type=Path, help="original image root; default data-root/images")
    parser.add_argument("--feature-ids", type=Path, help="original cached feature IDs; default features/clip_vitl14_ids.npy")
    parser.add_argument("--model-revision", required=True, help="record the downloaded official model's commit/revision")
    parser.add_argument("--variant", default=None, help="local weight variant, e.g. fp16 when that variant was downloaded")
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strength", type=float, default=.18)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--min-structure", type=float, default=.4)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="reuse identical configuration; retry pending/errors and extend count")
    parser.add_argument("--check-data", action="store_true", help="validate selection/paths only; no model load or output writes")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    try:
        summary = run_generation(args)
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
        if summary.get("error", 0):
            raise SystemExit(1)
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
