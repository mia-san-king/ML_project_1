#!/usr/bin/env python3
"""Download SMAPshot thumbnails for splits that are actually used.

Skips outside and buffer. Priority: test sets and val, then 50k train, then the rest.
IIIF URLs that request a 200 px thumbnail are rewritten to 512 px. Existing files are skipped.
"""
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import requests
from PIL import Image

from lib.paths import IMAGES, META

UA = "EPFL CS-433 student project (contact: yijun.liu@epfl.ch)"
KEEP_FIRST = {"test_region": 0, "test_city": 0, "test_in": 0, "val": 0, "train": 1}


def img_path(image_id):
    image_id = int(image_id)
    return IMAGES / f"{image_id // 1000:04d}" / f"{image_id}.jpg"


def upgrade_url(url):
    return url.replace("/full/200,/", "/full/!512,512/")


def download_queue():
    meta = pd.read_parquet(META / "smapshot.parquet", columns=["id", "image_url"])
    splits = pd.read_csv("data/splits/main.csv.gz")
    frame = meta.merge(splits, on="id")
    frame["base"] = frame.split.str.replace(r"^nadir_", "", regex=True)
    frame = frame[~frame.base.isin(["outside", "buffer", "missing"])].copy()
    frame["pri"] = frame.base.map(KEEP_FIRST).astype(int)
    rng = np.random.default_rng(0)
    train_index = frame.index[frame.base == "train"]
    early = rng.choice(train_index.to_numpy(), size=min(50_000, len(train_index)), replace=False)
    frame.loc[train_index.difference(early), "pri"] = 2
    frame["tie"] = rng.random(len(frame))
    frame["image_url"] = frame.image_url.map(upgrade_url)
    return frame.sort_values(["pri", "tie"])


def main(delay=0.5, max_side=512):
    queue = download_queue()
    print(f"queue={len(queue)} delay={delay}", flush=True)
    session = requests.Session()
    session.headers["User-Agent"] = UA
    failed_path = IMAGES / "failed.txt"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed = open(failed_path, "a")
    done = 0
    for n, (image_id, url) in enumerate(zip(queue.id, queue.image_url), start=1):
        path = img_path(image_id)
        if path.exists():
            continue
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image.thumbnail((max_side, max_side))
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".jpg.part")
            image.save(partial, format="JPEG", quality=90)
            partial.replace(path)
            done += 1
        except Exception as exc:
            failed.write(f"{image_id}\t{url}\t{exc!r}\n")
            failed.flush()
        if n % 200 == 0:
            print(f"seen={n} saved_this_run={done} last={image_id}", flush=True)
        time.sleep(delay)
    failed.close()
    print(f"finished seen={len(queue)} saved_this_run={done}", flush=True)


if __name__ == "__main__":
    main(delay=float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)
