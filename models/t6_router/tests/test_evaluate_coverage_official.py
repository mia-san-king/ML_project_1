import json

import joblib
import numpy as np
import pandas as pd
import pytest

from models.t6_router import error_distribution_data as archived
from models.t6_router import evaluate_coverage_official as official
from models.t6_router.geographic_gate import NONPOSITION, POSITION
from scripts.confidence.evaluate import FEATURES, sha256


class Confidence:
    def __init__(self, value):
        self.value = value

    def predict_proba(self, raw):
        assert set(raw.columns) == set(FEATURES["T4"] + FEATURES["T5"])
        return np.full(len(raw), self.value)


class Gate:
    def predict_t4_probability(self, x):
        assert tuple(x.columns) == POSITION + NONPOSITION
        return x.T4_p25.to_numpy()


def fixture(root, monkeypatch, fault=None):
    ids = [1, 12, 31, 42, 53]
    splits = ["train", "val", *archived.TEST_SPLITS]
    source = pd.DataFrame({"id": ids, "split": splits, "block": range(5),
                           "latitude": [46.5] * 5, "longitude": [7.5] * 5})
    files = {}
    def dest(key):
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        files[key] = path
        return path
    source[["id", "split", "block"]].to_csv(dest("data/splits/main.csv.gz"), index=False)
    source[["id", "latitude", "longitude"]].to_parquet(dest("data/t1_original/meta/smapshot.parquet"), index=False)
    np.save(dest("data/t1_original/features/clip_vitl14_ids.npy"), ids)
    preds, raw = [], []
    for i, model in enumerate(("T1", "T3", "T4", "T5")):
        rows = source.iloc[1:].rename(columns={"latitude": "lat_pred", "longitude": "lon_pred"}).copy()
        rows["model"] = model
        rows["p_within_25km"] = .99
        rows["confidence_partition"] = np.where(rows.split.eq("val"), "fit", "test")
        rows["error_km"] = 0.
        if model == "T4":
            rows["lat_pred"] += 1
            rows["error_km"] = 6371.0088 * np.pi / 180.
        preds.append(rows)
        item = rows.drop(columns=["p_within_25km", "confidence_partition"]).copy()
        item[FEATURES[model]] = i + 1.
        if model == "T5" and fault == "block":
            item.loc[item.split.eq("test_region"), "block"] = 1234
        raw.append(item)
    prefix = "data/local_runs/confidence_models/"
    pd.concat(preds).sample(frac=1, random_state=5).to_parquet(dest(prefix + "predictions.parquet"), index=False)
    pd.concat(raw).sample(frac=1, random_state=6).to_parquet(dest(prefix + "inputs.parquet"), index=False)
    manifest = root / "assets.tsv"
    pd.DataFrame([{"root": "workspace", "path": k, "sha256": sha256(v)} for k, v in files.items()]).to_csv(
        manifest, sep="\t", index=False)
    monkeypatch.setattr(archived, "ARCHIVE", manifest)
    monkeypatch.setattr(official, "ARCHIVE", manifest)
    monkeypatch.setattr(archived, "EXPECTED", dict.fromkeys(archived.TEST_SPLITS, 1))
    gate_dir = root / "coverage/gate"
    gate_dir.mkdir(parents=True)
    hashes = {}
    for arm, p4 in [("three_regions", .9), ("seven_regions", .5)]:
        path = gate_dir / f"{arm}.joblib"
        joblib.dump({"gate": Gate(), "confidence": {"T4": Confidence(p4), "T5": Confidence(.1)}}, path)
        hashes[path.name] = sha256(path)
    (gate_dir / "summary.json").write_text(json.dumps({"outputs_sha256": hashes}))
    if fault == "gate":
        (gate_dir / "seven_regions.joblib").write_bytes(b"replacement")
    if fault == "raw":
        files[prefix + "inputs.parquet"].write_bytes(b"replacement")
    return gate_dir


@pytest.mark.parametrize("fault", ["gate", "raw", "block"])
def test_rejects_unverified_gate_raw_inputs_or_unpaired_blocks(tmp_path, monkeypatch, fault):
    gate = fixture(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError, match="digest|hash|align"):
        official.run(tmp_path, gate, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_all_official_tests_with_arm_specific_calibration_and_ties_to_t5(tmp_path, monkeypatch):
    gate = fixture(tmp_path, monkeypatch)
    output = tmp_path / "out"
    result = official.run(tmp_path, gate, output)
    assert result == json.loads((output / "summary.json").read_text())
    assert result["selected_model"] == "seven_regions"
    assert set(result["splits"]) == set(archived.TEST_SPLITS)
    decisions = pd.read_parquet(output / "decisions.parquet")
    assert decisions.id.tolist() == [31, 42, 53]
    assert decisions.three_regions_choose_T4.all() and not decisions.seven_regions_choose_T4.any()
    assert (decisions.seven_regions_T4_p25 == .5).all()
    assert (decisions.seven_regions_lat_pred == decisions.T5_lat_pred).all()
    assert (decisions.seven_regions_lon_pred == decisions.T5_lon_pred).all()
    for split in archived.TEST_SPLITS:
        methods = result["splits"][split]["methods"]
        assert methods["seven_regions"]["accuracy_non_ties_pct"] == 100
        assert methods["seven_regions"]["hits_25km"] == methods["T5"]["hits_25km"] == 1
        assert methods["three_regions"]["hits_25km"] == methods["T4"]["hits_25km"] == 0
        assert methods["seven_regions"]["median_km"] == methods["seven_regions"]["mean_km"] == 0.
        assert methods["seven_regions"]["histogram_counts"][0] == 1
    assert result["pooled"]["n"] == 3
    assert result["decisions_sha256"] == sha256(output / "decisions.parquet")
