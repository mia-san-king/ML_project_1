"""Compare nested SDXL doses with an equal-weight duplicated-parent T4 fit."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from experiment_paths import VOLUME
from unittest.mock import patch

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = "2"

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.clip_regression.scripts import train
from models.geoclip_time.scripts.rerank_references import _sha256

RUNS = ROOT / "data/local_runs/t3_t4_8192"
DATA = RUNS / "t4_source/augmented_view"
ARCHIVE = ROOT / "report/experiments/2026-11-20-diffusion-augmentation"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(pred, before):
    if not pred[["id", "split", "latitude", "longitude"]].equals(before[["id", "split", "latitude", "longitude"]]):
        raise ValueError("Query IDs or truth changed")
    result = {}
    for label, group in pred.groupby("split"):
        e = group.error_km
        if not np.isfinite(e).all():
            raise ValueError("Nonfinite evaluation error")
        result[label] = {"n":len(e), "within25":int((e <= 25).sum()),
                         "pct25":float(100*(e <= 25).mean()), "median_km":float(e.median())}
    return result


def main():
    torch.set_num_threads(2)
    manifest = read(VOLUME / "aug8192/t1_dataset/build_manifest.json")
    for name in ("features/clip_vitl14.npy", "features/clip_vitl14_ids.npy", "meta/smapshot.parquet", "splits/main.csv.gz"):
        if _sha256(DATA / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Input checksum mismatch: {name}")
    split = pd.read_csv(DATA / "splits/main.csv.gz").set_index("id", verify_integrity=True)
    ids = np.load(DATA / "features/clip_vitl14_ids.npy")
    real = np.sort(ids[(ids > 0) & split.loc[ids, "split"].eq("train").to_numpy()])
    ordered = np.random.default_rng(0).permutation(real)
    ordered = ordered[np.isin(-ordered, ids)]
    if len(real) != 176308 or len(ordered) != 8192:
        raise ValueError("Unexpected counts")
    archived = read(ARCHIVE / "augmented/build_manifest.json")
    np.testing.assert_array_equal(np.sort(-ordered[:512]), np.sort(archived["synthetic_ids"]))
    before = pd.read_parquet(RUNS / "original/T4/predictions.parquet").sort_values("id").reset_index(drop=True)
    after = pd.read_parquet(RUNS / "augmented/T4/predictions.parquet").sort_values("id").reset_index(drop=True)
    result = {"diffusion_0":metrics(before,before), "diffusion_8192":metrics(after,before)}
    original_fit = train.fit_linear
    for dose, duplicate in ((512,False), (2048,False), (4096,False), (8192,True)):
        key = f"{'duplicate_original' if duplicate else 'diffusion'}_{dose}"
        parents = ordered[:dose]

        def subset_fit(clip, dates, train_rows, targets):
            row_ids = ids[train_rows]
            real_positions = np.flatnonzero(row_ids > 0)
            lookup = pd.Index(row_ids)
            chosen = lookup.get_indexer(parents if duplicate else -parents)
            if np.any(chosen < 0):
                raise ValueError("Missing parent or variant")
            np.testing.assert_array_equal(targets[lookup.get_indexer(parents)], targets[lookup.get_indexer(-parents)])
            positions = np.concatenate([real_positions, chosen])
            return original_fit(clip, dates, train_rows[positions], targets[positions])

        with tempfile.TemporaryDirectory() as tmp, patch.object(train, "fit_linear", subset_fit):
            output = Path(tmp) / "T4"
            fit = train.fit_and_predict(DATA, DATA / "splits/main.csv.gz", output)
            if fit["design_rank"] != 772:
                raise ValueError("Unexpected rank")
            pred = pd.read_parquet(output / "predictions.parquet").sort_values("id").reset_index(drop=True)
            result[key] = metrics(pred, before)
        print(key, json.dumps(result[key], sort_keys=True), flush=True)
    for label, remote in read(ARCHIVE / "models/T4/summary.json")["splits"].items():
        np.testing.assert_allclose([result["diffusion_512"][label]["median_km"],result["diffusion_512"][label]["pct25"]],
                                   [remote["median_km"], remote["acc@25km"]], atol=1e-7, rtol=0)
    report = {"real_train":len(real), "eval_n":len(before), "method":"Official float64 pivoted-QR T4. Seed-0 parent permutation prefixes, one variant per parent. Matched original-duplication control at 8192.",
              "parent_order_sha256":hashlib.sha256(ordered.astype("<i8").tobytes()).hexdigest(),
              "counts_0_and_8192_from":"completed local T4 predictions", "results":result}
    output = RUNS / "t4_augmentation_curve.json"
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(output)


if __name__ == "__main__":
    main()
