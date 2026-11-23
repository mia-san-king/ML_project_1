import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from models.t6_router.coverage_gate import ERRORS, RAW, SEVEN, USED, _signature
from models.t6_router.expanded_regions import SEVEN as REGIONS, matched_samples
from models.t6_router.geographic_gate import POSITION
from models.t7_error_regression import run_learning_curve as driver
from scripts.confidence.evaluate import sha256


def test_curve_reads_only_seven_dev_folds_and_checks_original_sample(tmp_path, monkeypatch):
    root = tmp_path / "study"
    (root / "splits").mkdir(parents=True)
    folds, frames = {}, []
    for i, name in enumerate(REGIONS):
        frame = pd.DataFrame({"id": np.arange(10*i+1, 10*i+4), "region": name, "block": i})
        frame[list(POSITION)] = [46., 7., 47., 8.]
        frame[RAW] = 1.
        frame[ERRORS] = [15., 50.]
        frames.append(frame)
        split = root / "splits" / f"{name}.csv.gz"
        frame.assign(split="query")[["id", "split", "block"]].to_csv(split, index=False)
        directory = root / name / "experts"
        directory.mkdir(parents=True)
        frame.to_parquet(directory / "predictions.parquet", index=False)
        receipt = {"inputs": {"split_sha256": sha256(split)}, "query_count": len(frame),
                   "predictions_sha256": sha256(directory / "predictions.parquet")}
        (directory / "summary.json").write_text(json.dumps(receipt))
        folds[name] = receipt
    sample = matched_samples(pd.concat(frames), budget=9)["seven"]
    counts = sample.region.value_counts().to_dict()
    digest = _signature(sample, USED)
    ids = _signature(sample, ["id"])
    t6_path, t7_path = (tmp_path / f"{prefix}.json" for prefix in ("t6", "t7"))
    t6 = {"folds": folds, "sampling_seed": 42, "comparison": {"development_n_per_arm": 9, "training": {SEVEN: {
        "sample_sha256": digest, "gate_train_ids_sha256": ids, "gate_train_n": 9}}}}
    t7 = {"training": {"n": 9, "photos_by_region": counts}, "inputs": {"training_sample_sha256": digest}}
    t6_path.write_text(json.dumps(t6))
    t7_path.write_text(json.dumps(t7))
    monkeypatch.setattr(driver, "T6_REPORT", t6_path)
    monkeypatch.setattr(driver, "T7_REPORT", t7_path)
    data, source = driver.load_development(root)
    assert data.id.tolist() == sample.id.tolist()
    assert source["training_ids_sha256"] == ids
    assert set(source["folds"]) == set(REGIONS)
    learning_curve = SimpleNamespace(evaluate_curve=lambda train: {"ids": train.id.tolist()})
    monkeypatch.setitem(sys.modules, "models.t7_error_regression.learning_curve", learning_curve)
    out = tmp_path / "report/curve.json"
    result = driver.run(root, out)
    assert result == json.loads(out.read_text()) and result["curve"]["ids"] == sample.id.tolist()
    original = out.read_bytes()
    with pytest.raises(FileExistsError):
        driver.run(root, out)
    assert out.read_bytes() == original
    t6["comparison"]["training"][SEVEN]["gate_train_ids_sha256"] = "wrong"
    t6_path.write_text(json.dumps(t6))
    with pytest.raises(ValueError, match="IDs"):
        driver.load_development(root)
    t6["comparison"]["training"][SEVEN]["gate_train_ids_sha256"] = ids
    t6_path.write_text(json.dumps(t6))
    t7["inputs"]["training_sample_sha256"] = "wrong"
    t7_path.write_text(json.dumps(t7))
    with pytest.raises(ValueError, match="signature"):
        driver.load_development(root)
