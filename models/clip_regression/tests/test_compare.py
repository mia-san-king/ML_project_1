"""Pairing and metric checks for the standalone T4/original-T1 comparison."""
import importlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def comparison():
    name = "models.clip_regression.scripts.compare_t1"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "T4/T1 comparison is not implemented"
    return importlib.import_module(name)


def frames():
    # At the equator, each longitude radian is exactly one Earth radius.
    degrees_per_km = 180 / (np.pi * 6371.0088)
    t4 = pd.DataFrame({
        "id": [11, 12, 13, 14, 15],
        "split": ["test_in"] * 4 + ["val"],
        "latitude": [0.] * 5, "longitude": [0.] * 5,
        "lat_pred": [0.] * 5,
        "lon_pred": np.array([10., 30., 40., 0., 0.]) * degrees_per_km,
        "error_km": [10., 30., 40., 0., 0.],
    })
    t1 = t4[["id", "split", "lat_pred", "lon_pred"]].copy()
    t1["lon_pred"] = np.array([30., 10., 40., 20., 0.]) * degrees_per_km
    return t4, t1.iloc[[3, 1, 4, 0, 2]].reset_index(drop=True)


def test_pairs_by_id_and_reports_distance_and_hit_differences():
    t4, t1 = frames()
    result = comparison().compare_predictions(t4, t1)
    assert result["alignment"]["n"] == 5
    row = result["by_split"]["test_in"]
    assert row["n"] == 4
    assert row["t4"]["median_km"] == pytest.approx(20.)
    assert row["t4"]["mean_km"] == pytest.approx(20.)
    assert row["t1"]["median_km"] == pytest.approx(25.)
    assert row["t1"]["mean_km"] == pytest.approx(25.)
    assert row["t4"]["acc@25km"] == row["t1"]["acc@25km"] == 50.
    assert row["t4_minus_t1"]["mean_km"] == pytest.approx(-5.)
    assert row["t4_minus_t1"]["hits@25km"] == 0
    assert row["paired"] == {"t4_closer": 2, "t4_farther": 1, "equal_error": 1,
                             "t4_gained_hit25": 1, "t4_lost_hit25": 1,
                             "both_hit25": 1, "both_missed25": 1}
    assert result["by_split"]["val"]["paired"]["equal_error"] == 1


@pytest.mark.parametrize("change,match", [
    ("missing", "ID sets"), ("duplicate", "duplicate"),
    ("split", "split mismatch"), ("null_id", "ID"),
    ("invalid_gps", "coordinates"), ("wrong_error", "error_km"),
    ("postprocessed", "original T1 grid"), ("train", "split"),
])
def test_rejects_noncomparable_or_corrupt_predictions(change, match):
    t4, t1 = frames()
    if change == "missing":
        t1 = t1.iloc[1:]
    elif change == "duplicate":
        t1 = pd.concat([t1, t1.iloc[:1]])
    elif change == "split":
        t1.loc[0, "split"] = "test_city"
    elif change == "null_id":
        t4.loc[0, "id"] = np.nan
    elif change == "invalid_gps":
        t1.loc[0, "lat_pred"] = 91.
    elif change == "wrong_error":
        t4.loc[0, "error_km"] += 0.01
    elif change == "postprocessed":
        t1["method"] = "year"
    elif change == "train":
        t4.loc[0, "split"] = "train"
    with pytest.raises(ValueError, match=match):
        comparison().compare_predictions(t4, t1)


def test_external_truth_supports_minimal_schema_and_rejects_conflicting_truth():
    t4, t1 = frames()
    truth = t4[["id", "latitude", "longitude"]].iloc[::-1]
    minimal = t4.drop(columns=["latitude", "longitude"])
    assert comparison().compare_predictions(minimal, t1, truth=truth)["alignment"]["n"] == 5
    conflict = truth.copy()
    conflict.loc[0, "latitude"] = 1.
    with pytest.raises(ValueError, match="truth"):
        comparison().compare_predictions(t4, t1, truth=conflict)
    with pytest.raises(ValueError, match="truth|data-root"):
        comparison().compare_predictions(minimal, t1)


def test_cli_runs_outside_repository_and_saves_json(tmp_path):
    module = comparison()
    t4, t1 = frames()
    t4_path, t1_path, output = [tmp_path / name for name in ("t4.parquet", "t1.parquet", "comparison.json")]
    t4.to_parquet(t4_path, index=False)
    t1.to_parquet(t1_path, index=False)
    cmd = [sys.executable, module.__file__, "--t4", str(t4_path), "--t1", str(t1_path), "--output", str(output)]
    run = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads(output.read_text())
    assert report["by_split"]["test_in"]["paired"]["t4_closer"] == 2
    assert report["inputs"]["t1"]["path"] == str(t1_path)
    assert len(report["inputs"]["t1"]["sha256"]) == 64
    assert "test_in" in run.stdout
    t1.iloc[:-1].to_parquet(t1_path, index=False)
    output.unlink()
    failed = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert failed.returncode != 0 and "ID sets" in failed.stderr
    assert not output.exists()
