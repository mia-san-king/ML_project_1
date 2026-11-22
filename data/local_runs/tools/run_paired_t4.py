"""Replay the project's closed-form T4 control with and without 8,192 variants."""

import hashlib
import json
import os
from pathlib import Path
from experiment_paths import VOLUME
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "data/local_runs/t3_t4_8192"
SOURCE = RESULTS / "t4_source"
ORIGINAL = ROOT / "data/t1_original"
AUGMENTED = VOLUME / "aug8192/t1_dataset"
ARCHIVE = ROOT / "report/experiments/2026-11-20-clip-regression-t4/summary.json"
os.environ["P2_DATA"] = str(SOURCE / "geospatial_cache")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from lib import geo
from models.clip_regression.scripts import train


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(source, target):
    target.parent.mkdir(exist_ok=True, parents=True)
    if target.exists():
        if sha(source) != sha(target):
            raise ValueError(f"Existing view differs at {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    if sha(source) != sha(target):
        raise ValueError(f"View copy differs at {target}")


def boundary():
    archive = SOURCE / "ne_10m_admin_0_countries.zip"
    countries = gpd.read_file(archive)
    swiss = countries.loc[countries.ADM0_A3 == "CHE"]
    assert len(swiss) == 1
    path = SOURCE / "ne_admin0.gpkg"
    if not path.exists():
        swiss.to_file(path)
    restored = gpd.read_file(path)
    assert restored.geometry.iloc[0].equals_exact(swiss.geometry.iloc[0], 0)
    border_lv95 = restored.to_crs(2056).geometry.union_all()
    # Only the country polygon enters grid creation; it provides an archived
    # independent numeric fingerprint of the geometry and its reprojection.
    with patch.object(geo, "load_boundaries", return_value=(border_lv95, None)):
        regenerated = geo.swiss_grid()
    original_grid = np.load(ORIGINAL / "geo/grid_1km.npy")
    same_grid = bool(np.array_equal(regenerated, original_grid))
    if not same_grid:
        if regenerated.shape != original_grid.shape or not np.allclose(regenerated, original_grid, rtol=0, atol=1e-5):
            raise RuntimeError("The downloaded boundary fails the archived grid check")
    audit = {"source": geo.NE + archive.name, "source_sha256": sha(archive), "filter": "ADM0_A3 == CHE",
             "grid_shape": list(original_grid.shape), "grid_bitwise_equal": same_grid,
             "grid_max_abs_deg": float(np.max(np.abs(regenerated - original_grid))),
             "archived_grid_sha256": sha(ORIGINAL / "geo/grid_1km.npy"),
             "boundary_geometry_wgb_sha256": hashlib.sha256(restored.geometry.iloc[0].wkb).hexdigest()}
    return path, audit


def execute(label, data, split, polygon):
    view = SOURCE / (label + "_view")
    for item in ("features/clip_vitl14.npy", "features/clip_vitl14_ids.npy", "meta/smapshot.parquet"):
        stage(data / item, view / item)
    stage(polygon, view / "geo/ne_admin0.gpkg")
    stage(split, view / "splits/main.csv.gz")
    output = RESULTS / label / "T4"
    receipt = RESULTS / label / "t4_timing.json"
    if output.exists() and receipt.exists():
        return json.loads(receipt.read_text()), pd.read_parquet(output / "predictions.parquet")
    fit_seconds = []
    solver = train.fit_linear

    def timed_solver(*args, **kwargs):
        start = time.perf_counter()
        result = solver(*args, **kwargs)
        fit_seconds.append(time.perf_counter() - start)
        return result

    start = time.perf_counter()
    with patch.object(train, "fit_linear", timed_solver):
        info = train.fit_and_predict(view, view / "splits/main.csv.gz", output)
    elapsed = time.perf_counter() - start
    pred = pd.read_parquet(output / "predictions.parquet")
    assert pred.id.is_unique and pred.id.gt(0).all() and len(pred) == 93538
    assert info["train_n"] == (176308 if label == "original" else 184500)
    assert info["design_rank"] == 772 and len(fit_seconds) == 1
    report = {"device": "cpu", "threads": torch.get_num_threads(), "torch": str(torch.__version__),
              "fit_seconds": fit_seconds[0], "fit_and_predict_seconds": elapsed,
              "summary": str(output / "summary.json"), "train_n": info["train_n"]}
    if label == "original":
        archived = json.loads(ARCHIVE.read_text())
        diffs = {group: {key: value - archived["splits"][group][key] for key, value in scores.items()}
                 for group, scores in info["splits"].items()}
        assert info["train_id_sha256"] == archived["train_id_sha256"]
        report["local_minus_archived_metrics"] = diffs
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report, pred


def main():
    torch.set_num_threads(2)
    polygon, source_info = boundary()
    print(json.dumps(source_info), flush=True)
    base, old = execute("original", ORIGINAL, ROOT / "data/splits/main.csv.gz", polygon)
    augmented, new = execute("augmented", AUGMENTED, AUGMENTED / "splits/main.csv.gz", polygon)
    old, new = (frame.set_index("id").sort_index() for frame in (old, new))
    assert old[["split", "latitude", "longitude"]].equals(new[["split", "latitude", "longitude"]])
    measurements = {}
    for group, real in old.groupby("split"):
        added = new.loc[real.index]
        before, after = real.error_km, added.error_km
        measurements[group] = {
            "n": len(before), "original_median_km": float(before.median()), "augmented_median_km": float(after.median()),
            "median_gain_km": float(before.median() - after.median()), "original_acc25_pct": float(100*(before <= 25).mean()),
            "augmented_acc25_pct": float(100*(after <= 25).mean()),
            "net_hits_25km": int((after <= 25).sum() - (before <= 25).sum())}
    report = {"boundary": source_info, "original": base, "augmented": augmented, "paired_metrics": measurements}
    (RESULTS / "t4_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(measurements), flush=True)


if __name__ == "__main__":
    main()
