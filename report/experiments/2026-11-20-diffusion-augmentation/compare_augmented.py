"""Compare predictions on the exact original queries and verify the baseline rerun inputs."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def read_predictions(path, model):
    frame = pd.read_parquet(path)
    if model == "T3":
        frame = frame[frame.method.eq("year")]
    return frame.set_index("id", verify_integrity=True).sort_index()


def error_km(frame, truth):
    lat, lon = np.deg2rad(truth[["latitude", "longitude"]].to_numpy(dtype=float)).T
    py, px = np.deg2rad(frame[["lat_pred", "lon_pred"]].to_numpy(dtype=float)).T
    h = np.sin((py-lat)/2)**2 + np.cos(lat)*np.cos(py)*np.sin((px-lon)/2)**2
    return 12742.0176 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def metrics(e):
    return {"median_km": float(np.median(e)), "mean_km": float(e.mean()),
            "within25": int((e<=25).sum()), "pct25": float(np.mean(e<=25)*100)}


def compare(args):
    root = Path(__file__).resolve().parent.parent
    original = {
        "T1": root/"2026-11-18-geoclip-time/preds/zhang7_T1_dual_time_available_s0_20261118.parquet",
        "T3": root/"2026-11-18-geoclip-t0-photo-rerank/results/predictions.parquet",
        "T4": root/"2026-11-20-clip-regression-t4/predictions.parquet",
    }
    truth = read_predictions(original["T4"], "T4")
    rows = []
    for model, path in original.items():
        before = read_predictions(path, model)
        after = read_predictions(getattr(args, model.lower()), model)
        for frame in (before, after):
            pd.testing.assert_index_equal(frame.index, truth.index)
            assert np.array_equal(frame.split, truth.split)
            assert len(frame) == 93538 and (frame.index > 0).all()
        old_e, new_e = error_km(before, truth), error_km(after, truth)
        assert np.isfinite(old_e).all() and np.isfinite(new_e).all()
        for split in ["val", "test_in", "test_city", "test_region"]:
            keep = truth.split.eq(split).to_numpy()
            old, new = metrics(old_e[keep]), metrics(new_e[keep])
            rows.append({"model": model, "split": split, "n": int(keep.sum()),
                         "original": old, "augmented": new,
                         "median_change_km": new["median_km"] - old["median_km"],
                         "hit_change": new["within25"] - old["within25"]})
    print(json.dumps(rows, indent=2))
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("t1","t3","t4","output"):
        parser.add_argument("--" + option, type=Path, required=True)
    compare(parser.parse_args())
