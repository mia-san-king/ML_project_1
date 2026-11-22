#!/usr/bin/env python3
"""CPU-only, post-completion comparison of the authorized T0/T1 run."""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

BASE = Path("/rcp-scratch/students/zhang7/ml-project2")
REPO = BASE / "ML_project_1"
DATA = BASE / "data"
RUN = BASE / "runs/2026-11-18-geoclip-time"
NAMES = {
    "T0": "zhang7_T0_no_time_available_s0_20261118",
    "T1": "zhang7_T1_dual_time_available_s0_20261118",
}
PIPELINE_LABELS = {"T0": "T0_no_time", "T1": "T1_dual_time"}
SPLITS = ("val", "test_in", "test_city", "test_region")
METRIC_KEYS = ("median_km", "mean_km", "acc@1km", "acc@5km", "acc@25km")
sys.path.insert(0, str(REPO))
from models.geoclip_time.lib.temporal_data import encode_date_intervals


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def haversine_errors(frame, label):
    """Independent NumPy great-circle calculation; all arrays share joined IDs."""
    true_lat = np.deg2rad(frame["latitude"].to_numpy(dtype=np.float64))
    true_lon = np.deg2rad(frame["longitude"].to_numpy(dtype=np.float64))
    pred_lat = np.deg2rad(frame[f"lat_pred_{label}"].to_numpy(dtype=np.float64))
    pred_lon = np.deg2rad(frame[f"lon_pred_{label}"].to_numpy(dtype=np.float64))
    a = (np.sin((pred_lat - true_lat) / 2) ** 2
         + np.cos(true_lat) * np.cos(pred_lat) * np.sin((pred_lon - true_lon) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def summarize(errors):
    require(len(errors) > 0 and np.isfinite(errors).all(), "empty/nonfinite error group")
    return {
        "median_km": float(np.median(errors)),
        "mean_km": float(np.mean(errors)),
        **{f"acc@{distance}km": float(np.mean(errors <= distance) * 100)
           for distance in (1, 5, 25)},
    }


def comparison_row(split, count, zero, one, **group):
    return {
        "split": split, "n": int(count), **group,
        "T0": zero, "T1": one,
        "delta_T1_minus_T0": {key: float(one[key] - zero[key]) for key in METRIC_KEYS},
    }


def aligned_predictions(path, label):
    frame = pd.read_parquet(path)
    required = {"id", "split", "lat_pred", "lon_pred"}
    require(required.issubset(frame), f"{label}: missing prediction columns")
    require(frame.id.notna().all() and not frame.id.duplicated().any(), f"{label}: duplicate/null prediction IDs")
    require(set(frame.split) == set(SPLITS), f"{label}: unexpected/missing prediction split")
    coordinates = frame[["lat_pred", "lon_pred"]].to_numpy(dtype=np.float64)
    require(np.isfinite(coordinates).all(), f"{label}: nonfinite predictions")
    require((np.abs(coordinates[:, 0]) <= 90).all() and (np.abs(coordinates[:, 1]) <= 180).all(), f"{label}: invalid predicted GPS")
    return frame[list(required)].sort_values(["split", "id"]).reset_index(drop=True)


def make_plots(joined, logs, overwrite):
    paths = [RUN / "training_curves.png", RUN / "test_in_error_cdf.png"]
    if not overwrite:
        require(not any(path.exists() for path in paths), "plot output exists; pass --overwrite to replace this comparison's outputs")
    with tempfile.TemporaryDirectory(prefix="geoclip-comparison-mpl-") as mpl_cache:
        os.environ["MPLCONFIGDIR"] = mpl_cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except ImportError as exc:
            return [], f"Optional plots skipped: matplotlib unavailable ({exc})"
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for label, color in (("T0", "#377eb8"), ("T1", "#e6550d")):
            history = logs[label]["history"]
            epochs = [row["epoch"] for row in history]
            axes[0].plot(epochs, [row["train_loss"] for row in history], label=label, color=color)
            axes[1].plot(epochs, [row["val_median_km"] for row in history], label=label, color=color)
        axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
        axes[1].set(title="Validation: fixed 256-image selection subset", xlabel="Epoch", ylabel="Median error (km)")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.suptitle("T0 vs T1: training curves; final evaluation uses all available cached queries")
        figure.tight_layout()
        figure.savefig(paths[0], dpi=150)
        plt.close(figure)

        test = joined.loc[joined.split.eq("test_in")]
        figure, axis = plt.subplots(figsize=(6.8, 4.5))
        for label, color in (("T0", "#377eb8"), ("T1", "#e6550d")):
            ordered = np.sort(test[f"error_{label}_km"].to_numpy())
            axis.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered) * 100, label=label, color=color)
        axis.axvline(25, color="gray", linestyle="--", linewidth=1)
        axis.set(title=f"test_in: all available cached queries (n={len(test):,})",
                 xlabel="Great-circle error (km)", ylabel="Queries at or below error (%)", ylim=(0, 100))
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(paths[1], dpi=150)
        plt.close(figure)
    return [str(path) for path in paths], None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace comparison.json and the two comparison plots only")
    args = parser.parse_args()
    require("torch" not in sys.modules, "analysis must not import torch")
    output = RUN / "comparison.json"
    require(args.overwrite or not output.exists(), "comparison.json exists; pass --overwrite to replace it")

    manifest = read_json(RUN / "input_manifest.json")
    status = read_json(RUN / "pipeline_status.json")
    require(status.get("state") == "completed", "pipeline has not completed; do not analyze partial results")
    current_commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    require(current_commit == manifest["code_commit"] == status["code_commit"], "repository, input manifest, and pipeline code commits differ")
    require(not subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True).strip(), "tracked repository files differ from the recorded commit")

    split_path = REPO / "data/splits/main.csv.gz"
    metadata_path = DATA / "meta/smapshot.parquet"
    ids_path = DATA / "features/clip_vitl14_ids.npy"
    manifest_by_path = {entry["path"]: entry for entry in manifest["inputs"]}
    checked_hashes = {}
    for path in (split_path, metadata_path, ids_path):
        entry = manifest_by_path.get(str(path))
        require(entry is not None, f"analysis input absent from manifest: {path}")
        actual_hash = sha256(path)
        require(actual_hash == entry["sha256"], f"analysis input changed since preflight: {path}")
        checked_hashes[str(path)] = actual_hash

    logs, saved, predictions = {}, {}, {}
    for label, name in NAMES.items():
        paths = {
            "training_log": DATA / "results/metrics" / f"{name}_train_log.json",
            "metrics": DATA / "results/metrics" / f"{name}_summary.json",
            "predictions": DATA / "results/preds" / f"{name}.parquet",
        }
        recorded = status["results"][PIPELINE_LABELS[label]]
        for key, path in paths.items():
            require(Path(recorded[key]).resolve() == path.resolve(), f"{label}: pipeline artifact path mismatch ({key})")
            require(path.is_file(), f"missing completed-run artifact: {path}")
        logs[label] = read_json(paths["training_log"])
        saved[label] = read_json(paths["metrics"])
        predictions[label] = aligned_predictions(paths["predictions"], label)
        require(logs[label]["run_name"] == name, f"{label}: training log run name mismatch")

    require(logs["T0"]["validation_ids"] == logs["T1"]["validation_ids"], "T0/T1 ordered validation IDs differ")
    validation_ids = logs["T0"]["validation_ids"]
    require(len(validation_ids) == 256 and len(set(validation_ids)) == 256, "expected 256 unique validation-selection queries")
    configs = {label: json.loads(json.dumps(logs[label]["config"])) for label in NAMES}
    for label, expected_mode in (("T0", "none"), ("T1", "both")):
        require(configs[label]["model"]["mode"] == expected_mode, f"{label}: unexpected model mode")
        del configs[label]["run_name"]
        del configs[label]["model"]["mode"]
    require(configs["T0"] == configs["T1"], "training configurations differ beyond run_name and temporal mode")
    time_config = logs["T0"]["time_config"]
    for label in NAMES:
        require(time_config == logs[label]["time_config"] == saved[label]["time_config"], f"{label}: inconsistent date encoding")
    pd.testing.assert_frame_equal(predictions["T0"][["id", "split"]], predictions["T1"][["id", "split"]], check_dtype=False)

    splits = pd.read_csv(split_path, usecols=["id", "split", "block"])
    require(splits.id.notna().all() and not splits.id.duplicated().any(), "invalid/duplicate split IDs")
    cached_ids = np.load(ids_path, allow_pickle=False)
    require(len(np.unique(cached_ids)) == len(cached_ids), "duplicate cached IDs")
    available = pd.DataFrame({"id": cached_ids}).merge(splits[["id", "split"]], on="id", how="left", validate="one_to_one")
    require(available.split.notna().all(), "cached ID absent from formal split")
    expected = available.loc[available.split.isin(SPLITS), ["id", "split"]].sort_values(["split", "id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(predictions["T0"][["id", "split"]], expected, check_dtype=False)
    require(set(validation_ids).issubset(set(expected.loc[expected.split.eq("val"), "id"])), "selection IDs are not all formal val queries")
    metadata = pd.read_parquet(metadata_path, columns=["id", "latitude", "longitude", "view_type", "date_shot_min", "date_shot_max"])
    require(metadata.id.notna().all() and not metadata.id.duplicated().any(), "invalid/duplicate metadata IDs")
    joined = predictions["T0"].merge(predictions["T1"], on=["id", "split"], validate="one_to_one", suffixes=("_T0", "_T1"))
    joined = joined.merge(splits[["id", "split", "block"]], on=["id", "split"], how="left", validate="one_to_one")
    joined = joined.merge(metadata, on="id", how="left", validate="one_to_one")
    coordinates = joined[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    require(np.isfinite(coordinates).all(), "prediction truth is missing/nonfinite")
    require((np.abs(coordinates[:, 0]) <= 90).all() and (np.abs(coordinates[:, 1]) <= 180).all(), "truth coordinates outside GPS bounds")
    for label in NAMES:
        joined[f"error_{label}_km"] = haversine_errors(joined, label)

    overall, verification = [], []
    for split in SPLITS:
        group = joined.loc[joined.split.eq(split)]
        primary = {}
        for label in NAMES:
            rows = [row for row in saved[label]["splits"] if row["split"] == split and row["view_type"] == "all"]
            require(len(rows) == 1, f"{label}/{split}: expected exactly one saved all-view metric row")
            row = rows[0]
            require(row["n"] == len(group), f"{label}/{split}: saved metric count mismatch")
            computed = summarize(group[f"error_{label}_km"].to_numpy())
            primary[label] = {key: float(row[key]) for key in METRIC_KEYS}
            np.testing.assert_allclose([primary[label][key] for key in METRIC_KEYS],
                                       [computed[key] for key in METRIC_KEYS], rtol=1e-7, atol=1e-6,
                                       err_msg=f"{label}/{split}: saved metrics disagree with ID-aligned independent Haversine")
            verification.append({"model": label, "split": split, "n": len(group),
                                 "absolute_differences": {key: abs(primary[label][key] - computed[key]) for key in METRIC_KEYS}})
        overall.append(comparison_row(split, len(group), primary["T0"], primary["T1"]))

    times, date_audit = encode_date_intervals(joined, **time_config)
    known = times[:, 2] > 0
    # Encoder endpoints are calendar years; round the float32 reconstruction to
    # the nearest half-year so exact 1930/1960/1990 boundaries stay in their bin.
    midpoint = np.rint((times[:, 0].astype(np.float64) * time_config["year_scale"] + time_config["year_center"]) * 2) / 2
    joined["view_group"] = joined.view_type.fillna("unknown").astype(str)
    joined["date_known"] = np.where(known, "known", "unknown")
    bins = np.full(len(joined), "unknown", dtype=object)
    bins[known & (midpoint < 1930)] = "<1930"
    bins[known & (midpoint >= 1930) & (midpoint < 1960)] = "1930-1959"
    bins[known & (midpoint >= 1960) & (midpoint < 1990)] = "1960-1989"
    bins[known & (midpoint >= 1990)] = ">=1990"
    joined["date_bin"] = bins
    exploratory = []
    for dimension in ("view_group", "date_known", "date_bin"):
        for (split, value), group in joined.groupby(["split", dimension], sort=True, dropna=False):
            zero = summarize(group.error_T0_km.to_numpy())
            one = summarize(group.error_T1_km.to_numpy())
            exploratory.append(comparison_row(split, len(group), zero, one, dimension=dimension, group=str(value)))
    plots, plot_note = make_plots(joined, logs, args.overwrite)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_names": NAMES,
        "provenance": {"code_commit": current_commit, "manifest": str(RUN / "input_manifest.json"),
                       "verified_analysis_input_sha256": checked_hashes,
                       "commit_check_scope": "Repository HEAD and completed pipeline status equal the input manifest; training logs do not independently embed a commit."},
        "metric_source": "Overall primary values are saved all-view metrics, cross-checked against independent ID-aligned Haversine (R=6371.0088 km).",
        "delta_definition": "T1 minus T0: negative km is better; positive accuracy percentage points is better.",
        "evaluation_scope": "All available cached val/test queries; not all originally assigned images. Missing-cache coverage is retained below.",
        "validation_selection": {"n": len(validation_ids), "same_ordered_ids": True, "ids": validation_ids,
                                 "note": "Training curves and checkpoint selection use only these 256 val images; final val/test metrics use all available cached queries."},
        "training": {label: {"best_epoch": logs[label]["epoch"], "best_val_subset_median_km": logs[label]["best_val_median_km"],
                             "history": logs[label]["history"], **{key: value for key, value in status["results"][PIPELINE_LABELS[label]].items()
                                                                    if key in ("training_seconds", "training_peak_mib", "evaluation_seconds")}}
                     for label in NAMES},
        "coverage": {label: saved[label]["audit"]["coverage"] for label in NAMES},
        "overall": overall, "metric_recomputation_checks": verification,
        "exploratory_note": "View/date subgroups are post-hoc descriptive analyses, not model selection or significance tests; small groups and spatial/temporal dependence matter. Date bins use interval midpoints, not exact capture dates.",
        "time_config": time_config, "evaluated_date_audit": date_audit,
        "exploratory_subgroups": exploratory, "plots": plots, "plot_note": plot_note,
    }
    require("torch" not in sys.modules, "analysis unexpectedly imported torch")
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output)
    print("split        n       T0median  T1median  delta_km    T0@25   T1@25  delta_pp")
    for row in overall:
        print(f"{row['split']:<11} {row['n']:>6}  {row['T0']['median_km']:>9.3f} {row['T1']['median_km']:>9.3f} {row['delta_T1_minus_T0']['median_km']:>9.3f}  {row['T0']['acc@25km']:>7.2f} {row['T1']['acc@25km']:>7.2f} {row['delta_T1_minus_T0']['acc@25km']:>9.2f}")
    print(f"Wrote {output}")
    for path in plots:
        print(f"Wrote {path}")
    if plot_note:
        print(plot_note)


if __name__ == "__main__":
    main()
