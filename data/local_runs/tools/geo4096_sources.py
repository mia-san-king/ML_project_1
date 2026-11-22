"""Stage only the frozen sparse training cohort from its public IIIF origins."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import io
import json
from pathlib import Path
from experiment_paths import VOLUME
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import requests
from PIL import Image

from fetch_public_augmentation_assets import group_photos, retry, session
from generate_more import writer_lock
from sdxl_recipe import _atomic_json, _now, _sha256

ROOT = Path(__file__).resolve().parents[3]
COHORT = ROOT / "data/local_runs/t4_geo4096"
STORE = VOLUME / "t4_geo4096"
PUBLIC = VOLUME / "public-assets"
EARLIER = VOLUME / "aug8192"


def selection():
    protocol = COHORT / "protocol.json"
    array = COHORT / "sparse_parents.npy"
    ids = np.load(array, allow_pickle=False)
    if (ids.ndim != 1 or len(ids) < 4096 or ids.dtype.kind not in "iu"
            or min(ids) <= 0 or len(ids) != len(set(map(int, ids)))):
        raise ValueError("Expected a predefined list of distinct positive parent IDs")
    split = pd.read_csv(ROOT / "data/splits/main.csv.gz", usecols=["id","split"]).set_index("id", verify_integrity=True)
    cached = np.load(ROOT / "data/t1_original/features/clip_vitl14_ids.npy", allow_pickle=False)
    if not split.loc[ids].split.eq("train").all() or not np.isin(ids, cached).all():
        raise ValueError("Only original cached training photos are permitted as parents")
    return ids.astype(np.int64), {"protocol_sha256":_sha256(protocol), "ordered_ids_file_sha256":_sha256(array)}


def old_parents():
    ids = np.load(EARLIER / "t1_dataset/features/clip_vitl14_ids.npy", allow_pickle=False)
    found = set(map(int, -ids[ids < 0]))
    if len(found) != 8192:
        raise ValueError("Expected previous 8192 accepted parents")
    return found


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pin(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing configuration differs: {path}")
    else:
        _atomic_json(path, value)


def run(store):
    ids, config = selection()
    old = old_parents()
    names = [f"images/{i//1000:04d}/{i}.jpg" for i in ids if int(i) not in old]
    groups = group_photos(names)
    store.mkdir(parents=True, exist_ok=True)
    with writer_lock(store / ".sources.lock"):
        pin(store / "source_config.json", config)
        receipts = read_json(PUBLIC / "public_assets_manifest.json")["files"]
        db_path = store / "sources.sqlite"
        with closing(sqlite3.connect(db_path)) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS sources(parent INTEGER PRIMARY KEY,status TEXT,path TEXT,sha TEXT,origin TEXT,reason TEXT)")
            db.commit()
        lock = threading.Lock()
        count = 0
        failures = []
        total = len(names)
        def group_work(group):
            nonlocal count
            with closing(sqlite3.connect(db_path, timeout=90)) as db, session() as client:
                for name, url in group:
                    parent = int(Path(name).stem)
                    cached = PUBLIC / name
                    target = store / "parents" / f"{parent}.jpg"
                    reason = None
                    record = db.execute("SELECT status,path,sha FROM sources WHERE parent=?", (parent,)).fetchone()
                    if record and record[0] == "ok":
                        target = Path(record[1])
                        if not target.is_file() or _sha256(target) != record[2]:
                            raise ValueError(f"Source JPEG changed for {parent}")
                    elif record and record[0] == "unavailable":
                        reason = "previously recorded permanent unavailability"
                    else:
                        if name in receipts and cached.is_file():
                            target = cached
                            sha = _sha256(target)
                            if sha != receipts[name]["sha256"]:
                                raise ValueError(f"Previously verified JPEG changed: {parent}")
                        else:
                            def get():
                                try:
                                    with client.get(url, timeout=(12,60)) as response:
                                        response.raise_for_status()
                                        return response.content
                                finally:
                                    time.sleep(.5)
                            try:
                                body = retry(get)
                            except requests.HTTPError as exc:
                                if exc.response is None or exc.response.status_code not in (403,404,410):
                                    raise
                                reason = f"HTTP {exc.response.status_code}"
                            if reason is None:
                                target.parent.mkdir(parents=True, exist_ok=True)
                                tmp = target.with_suffix(".jpg.part")
                                try:
                                    with Image.open(io.BytesIO(body)) as raw, raw.convert("RGB") as image:
                                        image.thumbnail((512,512))
                                        image.save(tmp, format="JPEG", quality=90)
                                except (OSError, ValueError) as exc:
                                    reason = f"Invalid public image: {exc}"
                                if reason is None:
                                    tmp.replace(target)
                                    sha = _sha256(target)
                        db.execute("INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?)",
                                   (parent, "unavailable" if reason else "ok", None if reason else str(target),
                                    None if reason else sha, url, reason))
                        db.commit()
                    with lock:
                        count += 1
                        if reason:
                            failures.append(parent)
                        if count % 100 == 0 or count == total:
                            _atomic_json(store / "source_status.json", {"completed":count,"total":total,
                                         "unavailable":failures, "updated":_now()})
                            print(f"Source photos processed {count}/{total}, unavailable {len(failures)}", flush=True)
        print(f"Downloading/verifying {total} new parents including reserves, across {len(groups)} public hosts.", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            work = [pool.submit(group_work, group) for group in groups]
            for result in as_completed(work):
                result.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=STORE, type=Path)
    run(parser.parse_args().store.resolve())
