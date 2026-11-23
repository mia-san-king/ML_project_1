import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from models.t6_router.coverage_gate import USED, _signature
from models.t6_router.expanded_regions import SEVEN
from models.t7_error_regression import run_sample_size as driver


@pytest.mark.parametrize("fault", [None, "mismatched", "coverage", "receipt", "duplicate", "synthetic"])
def test_only_verified_dev1_photos_extend_the_original_sample(tmp_path, monkeypatch, fault):
    matched = pd.DataFrame({key: np.ones(7) for key in USED}).assign(id=np.arange(1, 8), region=list(SEVEN))
    dev1 = pd.concat([matched.iloc[:1], matched.iloc[:1].assign(id=11), matched.iloc[:1].assign(id=13)], ignore_index=True)
    receipt = {"query_count": 3, "predictions_sha256": "verified"}
    archive = {"folds": {name: {"query_count": 1} for name in SEVEN}}
    archive["folds"]["dev1"] = receipt.copy()
    if fault == "mismatched":
        dev1.loc[0, "T4_error_km"] += 2.
    if fault == "coverage":
        archive["folds"]["dev3"]["query_count"] += 1
    if fault == "receipt":
        receipt["predictions_sha256"] = "changed"
    if fault == "duplicate":
        dev1.loc[2, "id"] = 11
    if fault == "synthetic":
        dev1.loc[2, "id"] = -13
    t6_path = tmp_path / "t6.json"
    t6_path.write_text(json.dumps(archive))
    monkeypatch.setattr(driver, "T6_REPORT", t6_path)
    monkeypatch.setattr(driver, "EXPECTED_SIZES", (7, 9, 2))
    base_provenance = {"training_sample_sha256": _signature(matched, USED), "training_ids_sha256": _signature(matched, ["id"])}
    monkeypatch.setattr(driver, "load_development", lambda root: (matched.copy(), base_provenance.copy()))
    def read_fold(root, name, split):
        assert name == "dev1" and split == root / "splits/dev1.csv.gz"
        return dev1.copy(), receipt.copy()
    monkeypatch.setattr(driver, "read_fold", read_fold)
    if fault:
        with pytest.raises(ValueError):
            driver.load_samples(tmp_path)
        return
    original, full, provenance = driver.load_samples(tmp_path)
    assert full.id.tolist() == [1, 2, 3, 4, 5, 6, 7, 11, 13] and original.id.tolist() == list(range(1, 8))
    assert provenance["added"]["photos_by_region"] == {"dev1": 2}
    assert provenance["full"]["sample_sha256"] == _signature(full, USED)
    core = SimpleNamespace(evaluate_sample_size=lambda a, b: {"n": [len(a), len(b)]})
    monkeypatch.setitem(sys.modules, "models.t7_error_regression.sample_size", core)
    report = tmp_path / "out/sample.json"
    result = driver.run(tmp_path, report)
    assert json.loads(report.read_text()) == result
    assert result["comparison"] == {"n": [7, 9]}
    with pytest.raises(FileExistsError):
        driver.run(tmp_path, report)
