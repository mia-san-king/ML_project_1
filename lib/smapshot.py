import json
import time
from pathlib import Path

import pandas as pd
import requests

from lib.paths import META

BASE = "https://smapshot.heig-vd.ch/api/v1"
UA = {"User-Agent": "EPFL CS-433 student project (contact: yijun.liu@epfl.ch)"}
KEEP = [
    "id",
    "original_id",
    "collection_id",
    "owner_id",
    "title",
    "view_type",
    "date_shot_min",
    "date_shot_max",
    "latitude",
    "longitude",
    "height",
    "width",
    "image_url",
]


def fetch_pages(page=1000, delay=1.0, max_pages=None):
    """Resume from pages already saved under META."""
    session = requests.Session()
    session.headers.update(UA)
    offset, n = 0, 0
    while max_pages is None or n < max_pages:
        dest = META / f"page_{offset:07d}.json"
        if not dest.exists():
            response = session.get(
                f"{BASE}/images",
                params={"limit": page, "offset": offset, "state[]": "validated"},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            if "rows" not in payload:
                raise RuntimeError(f"unexpected payload at offset {offset}: {payload.keys()}")
            partial = dest.with_suffix(".json.part")
            partial.write_text(json.dumps(payload))
            partial.replace(dest)
            print(f"saved {dest.name} rows={len(payload['rows'])}", flush=True)
            time.sleep(delay)
        rows = json.loads(dest.read_text())["rows"]
        n += 1
        if len(rows) < page:
            break
        offset += page
    return n


def pages_to_parquet():
    rows = [
        row
        for path in sorted(META.glob("page_*.json"))
        for row in json.loads(path.read_text())["rows"]
    ]
    frame = pd.json_normalize(rows).rename(columns={"media.image_url": "image_url"})
    missing = [column for column in KEEP if column not in frame.columns]
    if missing:
        raise RuntimeError(f"missing columns: {missing}")
    frame = frame[KEEP].drop_duplicates("id").reset_index(drop=True)
    out = META / "smapshot.parquet"
    frame.to_parquet(out, index=False)
    print(f"wrote {out} shape={frame.shape}", flush=True)
    return frame
