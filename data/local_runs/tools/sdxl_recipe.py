"""Unchanged cross-platform helpers extracted from the archived SDXL generator.

Source: report/experiments/2026-11-20-diffusion-augmentation/source/scripts/augmentation/generate.py.
The upstream top-level fcntl import prevents importing its image operations on Windows.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


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
