#!/usr/bin/env python3
"""Extract frozen CLIP features for downloaded images (resumable via shards)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from geoclip import GeoCLIP
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.features import encode_clip_pre_mlp, verify_pre_mlp_matches_forward
from lib.paths import FEATS
from scripts.download_images import img_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = PROJECT_ROOT / "data/splits/main.csv.gz"
DEFAULT_NAME = "clip_vitl14"


class ImageIdDataset(Dataset):
    def __init__(self, ids: np.ndarray):
        self.ids = np.asarray(ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = int(self.ids[index])
        path = img_path(image_id)
        return image_id, Image.open(path).convert("RGB")


def collate_images(image_encoder):
    def _collate(batch):
        ids = [item[0] for item in batch]
        images = [item[1] for item in batch]
        pixel_values = image_encoder.preprocess_image(images)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        return np.asarray(ids, dtype=np.int64), pixel_values

    return _collate


def list_downloaded_ids(split_csv: Path) -> np.ndarray:
    frame = pd.read_csv(split_csv, usecols=["id"])
    ids = []
    for image_id in frame.id.to_numpy():
        if img_path(int(image_id)).is_file():
            ids.append(int(image_id))
    return np.asarray(ids, dtype=np.int64)


def shard_dir(name: str) -> Path:
    path = FEATS / f"{name}_shards"
    path.mkdir(parents=True, exist_ok=True)
    return path


def shard_paths(name: str, shard_index: int) -> tuple[Path, Path]:
    root = shard_dir(name)
    return root / f"shard_{shard_index:05d}.npy", root / f"shard_{shard_index:05d}_ids.npy"


def merge_shards(name: str) -> tuple[Path, Path]:
    root = shard_dir(name)
    feat_files = sorted(root.glob("shard_*_ids.npy"))
    feat_files = [p for p in feat_files if "_ids" in p.name]
    if not feat_files:
        raise FileNotFoundError(f"no shards under {root}")

    all_feats, all_ids = [], []
    for ids_path in sorted(root.glob("shard_*_ids.npy")):
        stem = ids_path.name.replace("_ids.npy", "")
        feat_path = root / f"{stem}.npy"
        if not feat_path.is_file():
            raise FileNotFoundError(f"missing {feat_path}")
        all_feats.append(np.load(feat_path, mmap_mode="r"))
        all_ids.append(np.load(ids_path))

    feats = np.concatenate(all_feats, axis=0)
    ids = np.concatenate(all_ids, axis=0)
    if np.isnan(feats).any():
        raise ValueError("NaN in merged features")

    out_feat = FEATS / f"{name}.npy"
    out_ids = FEATS / f"{name}_ids.npy"
    np.save(out_feat, feats)
    np.save(out_ids, ids)
    return out_feat, out_ids


def write_meta(name: str, **fields) -> None:
    meta_path = FEATS / f"{name}_meta.json"
    payload = {}
    if meta_path.is_file():
        payload = json.loads(meta_path.read_text())
    payload.update(fields)
    meta_path.write_text(json.dumps(payload, indent=2))


@torch.no_grad()
def cache_shard(
    image_encoder,
    ids: np.ndarray,
    device: str,
    batch_size: int,
    workers: int,
    verify_first_batch: bool,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        ImageIdDataset(ids),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_images(image_encoder),
    )
    chunks, id_chunks = [], []
    first = verify_first_batch
    for batch_ids, pixel_values in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        if first:
            feats = verify_pre_mlp_matches_forward(image_encoder, pixel_values)
            first = False
        else:
            feats = encode_clip_pre_mlp(image_encoder, pixel_values)
        chunks.append(feats.half().cpu().numpy())
        id_chunks.append(batch_ids)
    return np.concatenate(chunks, axis=0), np.concatenate(id_chunks, axis=0)


def print_cosine_sanity(feats: np.ndarray, ids: np.ndarray) -> None:
    """Compare cosine sim of a random pair vs two rows with nearby ids (same shard)."""
    x = feats[: min(len(feats), 5000)].astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    i, j = 0, min(1, len(x) - 1)
    k, m = 0, min(len(x) - 1, 100)
    sim_near = float((x[i] * x[j]).sum())
    sim_far = float((x[k] * x[m]).sum())
    print(f"cosine_sanity near_ids=({ids[i]},{ids[j]})={sim_near:.3f} random={sim_far:.3f}", flush=True)


def run_cache(
    *,
    name: str,
    batch_size: int,
    workers: int,
    shard_size: int,
    device: str,
    limit: int | None,
    merge_only: bool,
) -> None:
    failed_path = FEATS / "failed_ids.txt"
    failed: set[int] = set()
    if failed_path.is_file():
        for line in failed_path.read_text().splitlines():
            line = line.strip()
            if line:
                failed.add(int(line))

    if merge_only:
        out_feat, out_ids = merge_shards(name)
        print(f"merged {out_feat} shape={np.load(out_feat, mmap_mode='r').shape}")
        print(f"ids {out_ids} count={len(np.load(out_ids))}")
        return

    all_ids = list_downloaded_ids(SPLIT_CSV)
    if limit is not None:
        all_ids = all_ids[:limit]
    print(f"images_to_encode={len(all_ids)} shard_size={shard_size}", flush=True)

    model = GeoCLIP().to(device)
    model.eval()
    enc = model.image_encoder

    n_shards = (len(all_ids) + shard_size - 1) // shard_size
    verify_once = True
    for shard_index in range(n_shards):
        feat_path, ids_path = shard_paths(name, shard_index)
        if feat_path.is_file() and ids_path.is_file():
            print(f"skip shard {shard_index}", flush=True)
            verify_once = False
            continue

        start = shard_index * shard_size
        end = min(start + shard_size, len(all_ids))
        shard_ids = all_ids[start:end]
        ok_ids = []
        for image_id in shard_ids:
            try:
                with Image.open(img_path(int(image_id))) as im:
                    im.verify()
                ok_ids.append(int(image_id))
            except Exception:
                failed.add(int(image_id))
        if not ok_ids:
            continue
        ok_ids = np.asarray(ok_ids, dtype=np.int64)

        print(f"shard {shard_index}/{n_shards} n={len(ok_ids)}", flush=True)
        feats, ids_out = cache_shard(
            enc,
            ok_ids,
            device,
            batch_size,
            workers,
            verify_first_batch=verify_once,
        )
        verify_once = False
        np.save(feat_path, feats)
        np.save(ids_path, ids_out)

    failed_path.write_text("\n".join(str(i) for i in sorted(failed)) + ("\n" if failed else ""))

    out_feat, out_ids = merge_shards(name)
    ids_arr = np.load(out_ids)
    feats = np.load(out_feat, mmap_mode="r")
    write_meta(
        name,
        n_images=int(len(ids_arr)),
        feat_dim=768,
        dtype="float16",
        split_csv=str(SPLIT_CSV),
    )
    print(f"done {out_feat} failed={len(failed)}", flush=True)
    if len(ids_arr) >= 4:
        print_cosine_sanity(feats, ids_arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0, help="0 avoids NFS dataloader temp issues")
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="smoke test on first N ids")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    run_cache(
        name=args.name,
        batch_size=args.batch_size,
        workers=args.workers,
        shard_size=args.shard_size,
        device=args.device,
        limit=args.limit,
        merge_only=args.merge_only,
    )


if __name__ == "__main__":
    main()