import json

import joblib
import numpy as np
import pandas as pd
import pytest

from models.t6_router.coverage_gate import ERRORS, USED, _signature
from models.t6_router.evaluate import TEST_SPLITS
from models.t6_router.expanded_regions import FINAL
from models.t6_router.geographic_gate import POSITION
from models.t7_error_regression import model, run_full as driver
from models.t7_error_regression.metrics import evaluate
from models.t7_error_regression.tests.test_run import rows
from scripts.confidence.evaluate import sha256


def setup_sources(tmp_path, monkeypatch):
    old_dir = tmp_path / "original"
    old_dir.mkdir()
    matched = rows(range(1, 5), "dev1", 5)
    full = pd.concat([matched, rows(range(5, 10), "dev1", 5).assign(T4_error_km=400.)], ignore_index=True)
    held = rows([15, 22], FINAL, 11).assign(latitude=46., longitude=7., T6_choose_T4=False)
    official = rows([102, 103, 104], "", 42).assign(latitude=46., longitude=7., T6_choose_T4=True, split=list(TEST_SPLITS))
    old_model = model.fit(matched)
    joblib.dump(old_model, old_dir / "T7.joblib")
    groups = pd.concat([held.assign(split=FINAL), official], ignore_index=True)
    archive = groups[["id", "block", *POSITION, *ERRORS, "T6_choose_T4", "split"]].join(model.predict(old_model, groups))
    for key in ("lat_pred", "lon_pred", "error_km"):
        archive[f"T7_{key}"] = np.where(archive.T7_choose_T4, archive[f"T4_{key}"], archive[f"T5_{key}"])
    archive["T6_error_km"] = np.where(archive.T6_choose_T4, archive.T4_error_km, archive.T5_error_km)
    archive.sample(frac=1, random_state=2).to_parquet(old_dir / "decisions.parquet", index=False)
    inputs = {"training_sample_sha256": _signature(matched, USED)}
    estimates = {"matched": {"training_sample_sha256": inputs["training_sample_sha256"]},
                 "full": {"n": len(full), "sample_sha256": _signature(full, USED)},
                 "added": {"n": len(full) - len(matched)}}
    original = {"training": {"regressor_parameters": model.REGRESSOR_PARAMS, "n": len(matched)},
                "new_final": evaluate(archive.loc[archive.split.eq(FINAL)]),
                "official": {"splits": {key: evaluate(archive.loc[archive.split.eq(key)]) for key in TEST_SPLITS},
                             "pooled": evaluate(archive.loc[archive.split.isin(TEST_SPLITS)])},
                "inputs": inputs, "output_sha256": {key: sha256(old_dir / key) for key in ("T7.joblib", "decisions.parquet")}}
    (old_dir / "summary.json").write_text(json.dumps(original))
    t7_report, size_report = tmp_path / "t7.json", tmp_path / "size.json"
    t7_report.write_text(json.dumps(original))
    size_report.write_text(json.dumps({"provenance": estimates}))
    monkeypatch.setattr(driver, "T7_REPORT", t7_report)
    monkeypatch.setattr(driver, "SAMPLE_SIZE_REPORT", size_report)
    monkeypatch.setattr(driver, "load_samples", lambda *_: (matched, full, estimates))
    monkeypatch.setattr(driver, "load_inputs", lambda *_: (matched, held, official, inputs))
    return old_dir, full, archive


def test_full_fit_frozen_baseline_and_three_paired_official_splits(tmp_path, monkeypatch):
    old, full, baseline = setup_sources(tmp_path, monkeypatch)
    out = tmp_path / "new"
    result = driver.run(tmp_path, tmp_path, tmp_path, old, out)
    assert result["training"]["n"] == len(full)
    assert result["training"]["regressor_parameters"]["max_iter"] == 80
    assert result["new_final"]["n"] == 2 and result["official"]["pooled"]["n"] == 3
    assert set(result["official"]["splits"]) == set(TEST_SPLITS)
    data = pd.read_parquet(out / "decisions.parquet").sort_values("id")
    assert data.id.tolist() == sorted(baseline.id)
    assert data.original_T7_choose_T4.all() and not data.T7_choose_T4.any()
    np.testing.assert_array_equal(data.T7_lat_pred, data.T5_lat_pred)
    assert (data.T5_expected_km == 50.).all()
    full_model = joblib.load(out / "T7_full.joblib")
    assert full_model.t4.n_iter_ == full_model.t5.n_iter_ == 80
    assert result["output_sha256"]["T7_full.joblib"] == sha256(out / "T7_full.joblib")
    with pytest.raises(FileExistsError):
        driver.run(tmp_path, tmp_path, tmp_path, old, out)


@pytest.mark.parametrize("fault", ["digest", "expected", "identity", "truth", "metrics", "overlap"])
def test_invalid_t7_comparator_or_training_overlap_never_publishes(tmp_path, monkeypatch, fault):
    old, full, frame = setup_sources(tmp_path, monkeypatch)
    report = json.loads(driver.T7_REPORT.read_text())
    if fault == "digest":
        (old / "T7.joblib").write_bytes(b"wrong")
    elif fault == "overlap":
        full.loc[len(full)-1, "id"] = frame.id.iloc[0]
    elif fault == "metrics":
        report["official"]["pooled"]["methods"]["T7"]["mean_km"] += 7.
    else:
        name = {"expected":"T4_expected_km", "identity":"id", "truth":"T5_error_km"}[fault]
        frame.loc[0, name] += 1
        frame.to_parquet(old / "decisions.parquet", index=False)
        report["output_sha256"]["decisions.parquet"] = sha256(old / "decisions.parquet")
    driver.T7_REPORT.write_text(json.dumps(report))
    (old / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        driver.run(tmp_path, tmp_path, tmp_path, old, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
