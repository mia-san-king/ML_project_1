import json

import numpy as np
import pandas as pd
import pytest

from models.t6_router.coverage_gate import ERRORS, RAW, SEVEN, USED, _signature
from models.t6_router.expanded_regions import FINAL, SEVEN as REGIONS, matched_samples
from models.t6_router.geographic_gate import POSITION
from models.t7_error_regression import run as driver
from scripts.confidence.evaluate import sha256


def rows(ids, region, block):
    result = pd.DataFrame({"id": ids, "region": region, "block": block})
    result[list(POSITION)] = [46., 6., 47., 7.]
    result[RAW] = 1.
    result[ERRORS] = [15., 50.]
    return result


def decisions(frame, *, cohort):
    out = frame[["id", cohort, "block", *POSITION, *ERRORS]].copy().iloc[::-1]
    out["seven_regions_p_T4"] = np.where(out.id % 2, .5, .91)
    out["seven_regions_choose_T4"] = out.seven_regions_p_T4.gt(.5)
    for col in ("lat_pred", "lon_pred", "error_km"):
        out[f"seven_regions_{col}"] = np.where(out.seven_regions_choose_T4, out[f"T4_{col}"], out[f"T5_{col}"])
    return out


@pytest.mark.parametrize("fault", [None, "id", "block", "T5_lon_pred", "T4_error_km", "choice", "selected"])
def test_archived_t6_decisions_align_by_photo_before_comparing(tmp_path, fault):
    frame = rows([11, 20, 37], FINAL, 22)
    saved = decisions(frame, cohort="region")
    if fault in ("id", "block", "T5_lon_pred", "T4_error_km"):
        saved.loc[saved.index[0], fault] += 1
    elif fault == "choice":
        saved.loc[saved.index[0], "seven_regions_choose_T4"] = True
    elif fault == "selected":
        saved.loc[saved.index[0], "seven_regions_error_km"] += 1
    path = tmp_path / "decisions.parquet"
    saved.to_parquet(path, index=False)
    if fault:
        with pytest.raises(ValueError, match="T6"):
            driver._with_t6(frame, path, sha256(path), cohort="region")
    else:
        result = driver._with_t6(frame, path, sha256(path), cohort="region")
        assert result.id.tolist() == [11, 20, 37] and result.T6_choose_T4.tolist() == [False, True, False]
        with pytest.raises(ValueError, match="digest"):
            driver._with_t6(frame, path, "wrong", cohort="region")


def test_t7_reuses_identical_seven_region_sample_and_verified_t6_for_both_cohorts(tmp_path, monkeypatch):
    def write(path, value):
        path.write_text(json.dumps(value))
    study, off, archive = (tmp_path / name for name in ("study", "official", "archive"))
    gate = study / "gate"
    for path in (gate, off, archive, study / "splits"):
        path.mkdir(parents=True)
    reports, dev = {}, []
    for i, region in enumerate((*REGIONS, FINAL), 1):
        frame = rows(np.arange(i*100, i*100+3), region, i)
        split = study / "splits" / f"{region}.csv.gz"
        frame[["id", "block"]].assign(split="query").to_csv(split, index=False)
        folder = study / region / "experts"
        folder.mkdir(parents=True)
        frame.to_parquet(folder / "predictions.parquet", index=False)
        reports[region] = {"inputs": {"split_sha256": sha256(split)}, "query_count": len(frame),
                           "predictions_sha256": sha256(folder / "predictions.parquet")}
        write(folder / "summary.json", reports[region])
        if region != FINAL:
            dev.append(frame)
        else:
            reserved = frame
    seven = matched_samples(pd.concat(dev), budget=9)["seven"]
    decisions(reserved, cohort="region").to_parquet(gate / "decisions.parquet", index=False)
    (gate / "seven_regions.joblib").write_bytes(b"old T6")
    gate_hashes = {name: sha256(gate / name) for name in ("decisions.parquet", "seven_regions.joblib")}
    receipt = {"development_n_per_arm": 9, "final_input_sha256": _signature(reserved, USED),
               "training": {SEVEN: {"sample_sha256": _signature(seven, USED)}}, "outputs_sha256": gate_hashes}
    write(gate / "summary.json", receipt)
    study_info = {"sampling_seed": 42, "folds": reports, "comparison": receipt}
    write(archive / "matched_region_coverage.json", study_info)
    write(study / "study.json", study_info)
    formal = rows([2001, 2002, 2003], "", 9).drop(columns="region")
    formal["split"] = ["test_in", "test_city", "test_region"]
    formal = formal.set_index("id")
    source_hashes = {"data/splits/main.csv.gz": "official archive"}
    monkeypatch.setattr(driver, "_official_inputs", lambda _: (formal.drop(columns=RAW), formal[RAW], source_hashes))
    decisions(formal.reset_index(), cohort="split").to_parquet(off / "decisions.parquet", index=False)
    off_info = {"selected_model": SEVEN, "input_sha256": {**source_hashes, "coverage_summary": sha256(gate / "summary.json"),
                "seven_regions.joblib": gate_hashes["seven_regions.joblib"]}, "decisions_sha256": sha256(off / "decisions.parquet")}
    write(off / "summary.json", off_info)
    write(archive / "coverage_official.json", off_info)
    monkeypatch.setattr(driver, "ARCHIVED_REPORTS", archive)
    train, held, official, provenance = driver.load_inputs(tmp_path, study, off)
    assert train.id.tolist() == seven.id.tolist()
    assert set(train.id).isdisjoint(held.id) and set(train.id).isdisjoint(official.id)
    assert set(RAW).issubset(official) and held.T6_choose_T4.sum() == 2
    assert provenance["training_sample_sha256"] == _signature(seven, USED)
    outcome = driver.run(tmp_path, study, off, tmp_path / "results")
    assert set(outcome["official"]["splits"]) == set(formal.split)
    exported = pd.read_parquet(tmp_path / "results/decisions.parquet")
    assert set(exported.id) == set(reserved.id) | set(formal.index)
    np.testing.assert_array_equal(exported.T7_lat_pred, np.where(exported.T7_choose_T4, exported.T4_lat_pred, exported.T5_lat_pred))
    assert outcome["output_sha256"]["T7.joblib"] == sha256(tmp_path / "results/T7.joblib")
    with monkeypatch.context() as changed:
        def wrong_sample(frame, **kwargs):
            sample = matched_samples(frame, **kwargs)["seven"].copy()
            sample.loc[0, "T4_error_km"] += 1.
            return {"seven": sample}
        changed.setattr(driver, "matched_samples", wrong_sample)
        with pytest.raises(ValueError, match="training or reserved"):
            driver.load_inputs(tmp_path, study, off)
    # Updating the independent production report cannot authorize different source inputs.
    off_info["decisions_sha256"] = "stale"
    write(off / "summary.json", off_info)
    with pytest.raises(ValueError, match="archived"):
        driver.load_inputs(tmp_path, study, off)
