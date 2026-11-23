import json

import joblib
import numpy as np
import pandas as pd
import pytest

from models.t6_router import coverage_gate
from scripts.confidence.evaluate import FEATURES, sha256


def photos(region, count, start):
    i = np.arange(count)
    frame = pd.DataFrame({
        "id": start + i, "region": region, "block": [f"{region}_{j % 3}" for j in i],
        "T4_lat_pred": 46. + i / 1000, "T4_lon_pred": 8., "T5_lat_pred": 47., "T5_lon_pred": 7.,
        "T4_error_km": np.where(i % 3 == 1, 60., 10.), "T5_error_km": np.where(i % 4 == 1, 5., 40.),
    })
    for k in FEATURES["T4"] + FEATURES["T5"]:
        frame[k] = (i + 1) / 100.
    return frame


def matched_rows():
    names = tuple("ABCDEFG")
    three = pd.concat([photos(r, 14, 100 * (i + 1)) for i, r in enumerate(names[:3])], ignore_index=True)
    seven = pd.concat([photos(r, 6, 100 * (i + 1)) for i, r in enumerate(names)], ignore_index=True)
    return three, seven, photos("Final", 12, 1000)


def test_matched_budgets_holdout_isolation_and_artifacts(tmp_path, monkeypatch):
    three, seven, final = matched_rows()
    # One extra tie in the seven-region arm leaves it with only 41 training labels.
    ix = seven.index[seven.region == "G"][0]
    seven.loc[ix, "T4_error_km"] = seven.loc[ix, "T5_error_km"]
    seen = []
    fit = coverage_gate.fit_confidence

    def capture(frame):
        seen.append((set(frame.region), len(frame)))
        return fit(frame)

    monkeypatch.setattr(coverage_gate, "fit_confidence", capture)
    out = tmp_path / "gate"
    report = coverage_gate.run_coverage_gate(three, seven, final, output_dir=out)
    assert seen == [
        *((set("ABC") - {held}, 28) for held in "ABC"), (set("ABC"), 42),
        *((set("ABCDEFG") - {held}, 28) for held in "ABCDEFG"), (set("ABCDEFG"), 42),
    ]
    assert report == json.loads((out / "summary.json").read_text())
    assert report["gate_train_n_per_arm"] == 41
    assert report["development_n_per_arm"] == 42
    assert report["oof_confidence_train_n_per_fold"] == 28
    assert report["n"] == 12
    decisions = pd.read_parquet(out / "decisions.parquet")
    assert report["outputs_sha256"]["decisions.parquet"] == sha256(out / "decisions.parquet")
    for arm in ("three_regions", "seven_regions"):
        assert report["methods"][arm]["n_non_ties"] == 12
        row = report["training"][arm]
        assert sum(row["photos_by_region"].values()) == 42
        assert row["final_confidence_train_n"] == 42
        fitted = joblib.load(out / f"{arm}.joblib")
        np.testing.assert_array_equal(fitted["gate"].predict_t4_probability(
            decisions.rename(columns={f"{arm}_T4_p25": "T4_p25", f"{arm}_T5_p25": "T5_p25"})),
            decisions[f"{arm}_p_T4"])
    paired = report["seven_vs_three"]
    assert paired["improved"] - paired["worsened"] == (
        report["methods"]["seven_regions"]["correct_non_ties"] - report["methods"]["three_regions"]["correct_non_ties"])
    by_block = report["final_by_block"]
    assert set(by_block) == set(final.block)
    assert all(b["n"] == 4 for b in by_block.values())
    assert sum(b["net"] for b in by_block.values()) == paired["net_correct_non_ties"]
    for block, rows in decisions.groupby("block"):
        s3, s7 = rows.three_regions_error_km, rows.seven_regions_error_km
        assert by_block[block]["improved"] == int((s7 < s3).sum())
        assert by_block[block]["worsened"] == int((s7 > s3).sum())

    # Changing evaluation labels or row order cannot change a choice for the same ID.
    changed = final.assign(T4_error_km=30., T5_error_km=30.).sample(frac=1, random_state=9)
    second = tmp_path / "other"
    updated = coverage_gate.run_coverage_gate(three.sample(frac=1, random_state=1),
                                             seven.sample(frac=1, random_state=2), changed, output_dir=second)
    assert updated["methods"]["seven_regions"]["accuracy_non_ties_pct"] is None
    cols = [f"{a}_{suffix}" for a in ("three_regions", "seven_regions") for suffix in ("p_T4", "choose_T4")]
    pd.testing.assert_frame_equal(decisions.set_index("id")[cols].sort_index(),
                                  pd.read_parquet(second / "decisions.parquet").set_index("id")[cols].sort_index())
    with pytest.raises(FileExistsError):
        coverage_gate.run_coverage_gate(three, seven, final, output_dir=out)


def test_rejects_unequal_counts_region_overlap_and_inconsistent_shared_photos(tmp_path):
    three, seven, final = matched_rows()
    for a, b, f in ((three.iloc[:-1], seven, final), (three, seven, final.assign(region="A")),
                    (three, seven, final.assign(block="F_0")), (three, seven, final.assign(id=three.id.iloc[:12].to_numpy())),
                    (three, seven.assign(block="one"), final), (three, seven.assign(T5_lon_pred=8.5), final)):
        with pytest.raises(ValueError):
            coverage_gate.run_coverage_gate(a, b, f, output_dir=tmp_path / "unused")
