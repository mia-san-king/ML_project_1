import os
from pathlib import Path

DATA = Path(
    os.environ.get("P2_DATA", "/rcp-scratch/students/yijliu/ml-project2/data")
).expanduser()
META, GEO, IMAGES = DATA / "meta", DATA / "geo", DATA / "images"
FEATS, CKPT, RESULTS = DATA / "features", DATA / "ckpt", DATA / "results"
for p in (META, GEO, IMAGES, FEATS, CKPT, RESULTS):
    p.mkdir(parents=True, exist_ok=True)
