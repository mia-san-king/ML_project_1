"""Public fallback for the pinned SDXL/CLIP snapshots and selected SMAPshot JPEGs."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests
from PIL import Image

from experiment_paths import ROOT, PUBLIC
from generate_more import writer_lock


SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
CLIP = "openai/clip-vit-large-patch14"
CLIP_REV = "32bd64288804d66eefd0ccbe215aa642df71cc41"
SDXL_REV = "462165984030d82259a11f4367a4eed129e94a7b"
CLIP_FILES = ("config.json", "preprocessor_config.json", "merges.txt", "vocab.json",
              "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json")
CLIP_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
CHUNK = 2**20
SAFE_CLIP_BYTES = 1710540580
SAFE_CLIP_SHA = "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"
IMAGE_HOSTS = {
    "smapshot.heig-vd.ch", "iiif.library.ethz.ch", "volare.vorarlberg.at",
    "www.e-rara.ch", "bso-iiif.swissartresearch.net", "www.e-manuscripta.ch",
    "iiif.vherens50.unil.ch",
}
UA = os.environ.get("CS433_DOWNLOAD_USER_AGENT", "EPFL CS-433 student project (historical photo geolocation)")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(data, indent=2), encoding="utf-8")
    partial.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            result.update(block)
    return result


def safe_destination(root, name):
    path = root.joinpath(*name.split("/"))
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Outside destination: {name}")
    return path


def plan(count):
    archived = ROOT / "report/experiments/2026-11-20-diffusion-augmentation"
    run = read_json(archived / "synthetic_store/generation_manifest.json")
    summary = read_json(archived / "synthetic_store/generation_summary.json")
    build = read_json(archived / "augmented/build_manifest.json")
    ids_path = ROOT / "data/t1_original/features/clip_vitl14_ids.npy"
    split_path = ROOT / "data/splits/main.csv.gz"
    if (digest(split_path).hexdigest() != run["input_sha256"]["split"]
            or digest(ids_path).hexdigest() != run["input_sha256"]["feature_ids"]):
        raise ValueError("Cached IDs or split differ from the archived experiment")
    splits = pd.read_csv(split_path, usecols=["id", "split"])
    eligible = np.intersect1d(np.load(ids_path, allow_pickle=False), splits.loc[splits.split.eq("train"), "id"])
    if len(eligible) != summary["eligible_train"] or not 512 <= count <= len(eligible):
        raise ValueError("Unexpected training population or photo count")
    parents = np.random.default_rng(0).permutation(eligible)[:count].astype("<i8")
    if hashlib.sha256(parents[:512].tobytes()).hexdigest() != summary["selection_ids_sha256"]:
        raise ValueError("The first 512 parent IDs differ from the archived selection")
    if run["model_revision"] != SDXL_REV or len(run["model_files"]) != 18:
        raise ValueError("Unexpected SDXL snapshot")
    if Path(build["configuration"]["clip_model"]).name != CLIP_REV:
        raise ValueError("Unexpected CLIP snapshot")
    images = [f"images/{i // 1000:04d}/{i}.jpg" for i in parents.tolist()]
    model_records = {"sdxl-fp16/" + row["path"]: row for row in run["model_files"]}
    return images, model_records, hashlib.sha256(parents.tobytes()).hexdigest()


def session():
    client = requests.Session()
    client.headers["User-Agent"] = UA
    return client


def retry(action):
    for attempt in range(5):
        try:
            return action()
        except requests.RequestException as exc:
            response = exc.response
            status = response.status_code if response is not None else 0
            if attempt == 4 or (400 <= status < 500 and status not in (408, 425, 429)):
                raise
            wait = 2 ** (attempt + 1)
            if response is not None:
                interval = response.headers.get("Retry-After", "")
                if interval.isdecimal():
                    wait = max(wait, int(interval))
                elif interval:
                    try:
                        wait = max(wait, (parsedate_to_datetime(interval) - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError, OverflowError):
                        pass
            time.sleep(wait)


def tree(client, repo, revision):
    def request():
        with client.get(f"https://huggingface.co/api/models/{repo}/tree/{revision}",
                        params={"recursive": "true"}, timeout=(12, 45)) as response:
            response.raise_for_status()
            return {e["path"]: e for e in response.json() if e["type"] == "file"}
    return retry(request)


def checks(path, info, expected=None):
    if path.stat().st_size != info["size"]:
        raise ValueError(f"Incorrect file length: {path}")
    sha = digest(path).hexdigest()
    if "lfs" in info:
        if sha != info["lfs"]["oid"]:
            raise ValueError(f"Incorrect HF large-file SHA-256: {path}")
    else:
        git = hashlib.sha1(f"blob {info['size']}\0".encode("ascii") + path.read_bytes()).hexdigest()
        if git != info["oid"]:
            raise ValueError(f"Incorrect Git object ID: {path}")
    if expected and sha != expected:
        raise ValueError(f"Incorrect archived SHA-256: {path}")
    return sha


class Report:
    def __init__(self, root, selection, which, photo_count):
        self.root = root
        self.lock = threading.Lock()
        self.stage = root / ".public_staging"
        self.stage.mkdir(exist_ok=True)
        self.files = {}
        self.source_receipts = {}
        for name in ("public_assets_manifest.json", "transfer_manifest.json"):
            path = root / name
            if path.is_file():
                self.source_receipts.update(read_json(path).get("files", {}))
        journal = root / "public_asset_receipts.jsonl"
        if journal.is_file():
            with journal.open(encoding="utf-8") as previous:
                for line in previous:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.source_receipts[row["name"]] = row
        self.journal = journal.open("a", encoding="utf-8")
        self.status = {"state":"running", "scope":which, "photo_count":photo_count,
                       "selection_sha256":selection, "processed_photos":0, "valid_photos":0,
                       "model_files":0, "photo_failures":{}, "model_error":None, "active_weight":None}
        self.last_write = 0.0
        self.save(force=True)

    def save(self, force=False):
        if force or time.monotonic() - self.last_write >= 5:
            self.status["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(self.root / "public_asset_status.json", self.status)
            self.journal.flush()
            print(f"photos {self.status['valid_photos']}/{self.status['photo_count']}; "
                  f"models verified {self.status['model_files']}; current {self.status['active_weight']}", flush=True)
            self.last_write = time.monotonic()

    def photo_error(self, name, exc):
        with self.lock:
            self.status["photo_failures"][name] = str(exc)
            self.status["processed_photos"] += 1
            self.save()

    def file(self, name, source, path, sha):
        entry = {"name":name, "bytes":path.stat().st_size, "sha256":sha, "source":source}
        with self.lock:
            self.files[name] = entry
            self.journal.write(json.dumps(entry, separators=(",",":")) + "\n")
            if name.startswith("images/"):
                self.status["valid_photos"] += 1
                self.status["processed_photos"] += 1
            else:
                self.status["model_files"] += 1
                self.status["active_weight"] = None
            self.save()

    def weight_progress(self, name, current, size):
        with self.lock:
            self.status["active_weight"] = {"name":name, "received":current, "size":size}
            self.save()

    def model_finished(self, future):
        error = future.exception()
        if error is not None:
            with self.lock:
                self.status["model_error"] = f"{type(error).__name__}: {error}"
                self.save(force=True)


def receive_model(client, report, repo, rev, name, relative, info, expected):
    path = safe_destination(report.root, name)
    if path.is_file():
        sha = checks(path, info, expected)
        report.file(name, f"existing verified {repo}@{rev}", path, sha)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = safe_destination(report.stage.resolve(), name + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    size = info["size"]
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{relative}"

    def download():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            return
        if offset > size:
            raise ValueError(f"Oversize partial file: {partial}")
        headers = {"Range":f"bytes={offset}-"} if offset else {}
        # A new query generates a fresh short-lived CDN redirect if a retry is needed.
        with client.get(url, params={"download":"true", "r":str(time.time_ns())},
                        headers=headers, stream=True, timeout=(15, 100)) as response:
            response.raise_for_status()
            if offset and response.status_code == 206:
                if not response.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected HTTP byte range")
            elif response.status_code == 200:
                offset = 0
            elif response.status_code != 206:
                raise ValueError(f"Unexpected HTTP status for {name}: {response.status_code}")
            with partial.open("ab" if offset else "wb") as output:
                for chunk in response.iter_content(CHUNK):
                    if not chunk:
                        continue
                    offset += len(chunk)
                    if offset > size:
                        raise ValueError(f"Source returned too many bytes: {name}")
                    output.write(chunk)
                    report.weight_progress(name, offset, size)
            if offset != size:
                raise requests.ConnectionError(f"Expected {size} bytes, received {offset} for {name}")
    retry(download)
    sha = checks(partial, info, expected)
    partial.rename(path)
    report.file(name, f"{repo}@{rev}", path, sha)


def fetch_models(report, pinned):
    with session() as client:
        sdxl = tree(client, SDXL, SDXL_REV)
        for name, expected in pinned.items():
            relative = expected["path"]
            info = sdxl[relative]
            if info["size"] != expected["bytes"]:
                raise ValueError(f"HF metadata differs from archive: {name}")
            receive_model(client, report, SDXL, SDXL_REV, name, relative, info, expected["sha256"])
        clip = tree(client, CLIP, CLIP_REV)
        if clip["model.safetensors"]["size"] != SAFE_CLIP_BYTES or clip["model.safetensors"]["lfs"]["oid"] != SAFE_CLIP_SHA:
            raise ValueError("CLIP revision metadata differs from verified model object")
        present = [f for f in CLIP_WEIGHTS if (report.root/"clip"/f).is_file()]
        # Existing archived pytorch_model.bin suffices. New transfers use safetensors.
        weights = present[0] if present else "model.safetensors"
        for relative in [*CLIP_FILES, weights]:
            info = clip[relative]
            receive_model(client, report, CLIP, CLIP_REV, "clip/"+relative, relative, info,
                          SAFE_CLIP_SHA if relative == "model.safetensors" else None)


def group_photos(names):
    rows = pd.read_parquet(ROOT / "data/t1_original/meta/smapshot.parquet", columns=["id","image_url"]).set_index("id")
    groups = defaultdict(list)
    for name in names:
        url = rows.at[int(Path(name).stem), "image_url"].replace("/full/200,/", "/full/!512,512/")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in IMAGE_HOSTS or parsed.username:
            raise ValueError(f"Unexpected source for {name}: {parsed.hostname}")
        groups[parsed.hostname].append((name,url))
    return sorted(groups.values(), key=len, reverse=True)


def fetch_photo_group(report, group):
    with session() as client:
        for name, url in group:
            path = safe_destination(report.root, name)
            try:
                if path.is_file():
                    with Image.open(path) as image:
                        if image.format != "JPEG" or max(image.size)>512 or image.mode!="RGB":
                            raise ValueError("Existing file differs from cached thumbnail format")
                        image.verify()
                    sha = digest(path).hexdigest()
                    known = report.source_receipts.get(name)
                    if known and (known["sha256"] != sha or known["bytes"] != path.stat().st_size):
                        raise ValueError("Existing file differs from its receipt")
                    source = known.get("source", "existing SSH receipt") if known else "existing JPEG without checksum receipt"
                    report.file(name, source, path, sha)
                    continue

                def download():
                    try:
                        with client.get(url, timeout=(12,60)) as response:
                            response.raise_for_status()
                            return response.content
                    finally:
                        # Each host has one worker. Respect the archive downloader's delay.
                        time.sleep(0.5)
                body = retry(download)
                partial = safe_destination(report.stage.resolve(), name + ".part")
                partial.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(io.BytesIO(body)) as raw:
                    with raw.convert("RGB") as image:
                        image.thumbnail((512,512))
                        image.save(partial, format="JPEG", quality=90)
                sha = digest(partial).hexdigest()
                path.parent.mkdir(parents=True, exist_ok=True)
                partial.rename(path)
                report.file(name, url, path, sha)
            except (ValueError,OSError,requests.RequestException) as exc:
                report.photo_error(name, exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=PUBLIC)
    parser.add_argument("--photo-count", type=int, default=8320)
    parser.add_argument("--photo-workers", type=int, choices=range(1,5), default=4)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and local selection, with no network or asset writes.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only-images", action="store_true")
    mode.add_argument("--only-models", action="store_true")
    args = parser.parse_args()
    images, pinned, selection = plan(args.photo_count)
    groups = [] if args.only_models else group_photos(images)
    scope = "images" if args.only_images else "models" if args.only_models else "all"
    print(json.dumps({"scope":scope, "selection_sha256":selection, "eligible_downloads":sum(map(len,groups)),
                      "source_groups":{urlsplit(g[0][1]).hostname:len(g) for g in groups},
                      "pinned_sdxl_files":len(pinned), "sdxl_revision":SDXL_REV, "clip_revision":CLIP_REV}),flush=True)
    if args.dry_run:
        return
    root = args.assets.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with writer_lock(root / "transfer.lock"):
        report = Report(root, selection, scope, 0 if args.only_models else args.photo_count)
        try:
            with ThreadPoolExecutor(max_workers=1) as models, ThreadPoolExecutor(max_workers=args.photo_workers) as photos:
                model = None if args.only_images else models.submit(fetch_models, report, pinned)
                if model:
                    model.add_done_callback(report.model_finished)
                work = [photos.submit(fetch_photo_group, report, g) for g in groups]
                for future in as_completed(work):
                    future.result()
            write_json(root/"public_assets_manifest.json", {"files":report.files, "selection_sha256":selection,
                       "sdxl_revision":SDXL_REV,"clip_revision":CLIP_REV,"scope":scope})
            if report.status["model_error"]:
                raise RuntimeError(report.status["model_error"])
            if groups and report.status["valid_photos"] < min(8192,args.photo_count):
                raise RuntimeError("Insufficient valid selected parents, see photo_failures")
            report.status["state"] = "complete"
        except BaseException as exc:
            report.status.update(state="failed",error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            report.save(force=True)
            report.journal.close()


if __name__ == "__main__":
    main()
