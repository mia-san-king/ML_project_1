"""Fit the 80-stage T7 on all seven development regions and evaluate the frozen refit."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from models.t6_router.coverage_gate import ERRORS
from models.t6_router.evaluate import TEST_SPLITS
from models.t6_router.geographic_gate import POSITION
from models.t7_error_regression.model import INPUT_COLUMNS, REGRESSOR_PARAMS, fit, predict
from models.t7_error_regression.run import load_inputs
from models.t7_error_regression.run_learning_curve import T7_REPORT
from models.t7_error_regression.run_sample_size import load_samples
from scripts.confidence.evaluate import sha256


SAMPLE_SIZE_REPORT = T7_REPORT.with_name("sample_size.json")
FORECASTS = ["T4_expected_km", "T5_expected_km"]
SELECTED = ["T7_lat_pred", "T7_lon_pred", "T7_error_km"]


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _decisions(model, rows):
    inputs = rows.loc[:, list(INPUT_COLUMNS)]
    predictions = predict(model, inputs)
    if not predictions.index.equals(rows.index):
        raise ValueError("T7 prediction rows differ from their input photos")
    columns = ["id", "split", "block", "latitude", "longitude", *POSITION, *ERRORS, "T6_choose_T4"]
    frame = rows[columns].join(predictions, validate="one_to_one")
    for key in ("lat_pred", "lon_pred", "error_km"):
        frame[f"T7_{key}"] = np.where(frame.T7_choose_T4, frame[f"T4_{key}"], frame[f"T5_{key}"])
    return frame


def _original_decisions(directory, published, rows):
    if _json(directory / "summary.json") != published:
        raise ValueError("Original T7 report differs from the archived experiment")
    for name in ("T7.joblib", "decisions.parquet"):
        if sha256(directory / name) != published["output_sha256"][name]:
            raise ValueError(f"Original T7 digest mismatch for {name}")
    saved = pd.read_parquet(directory / "decisions.parquet").set_index("id", verify_integrity=True).sort_index()
    reproduced = _decisions(joblib.load(directory / "T7.joblib"), rows).set_index("id", verify_integrity=True).sort_index()
    if not saved.index.equals(reproduced.index):
        raise ValueError("Original T7 photo IDs differ from the current evaluation")
    exact = ["split", "block", *POSITION, *ERRORS, "T6_choose_T4", "T7_choose_T4", *SELECTED]
    for column in exact:
        if not np.array_equal(saved[column].to_numpy(), reproduced[column].to_numpy()):
            raise ValueError(f"Original T7 field {column} disagrees with the reproduced prediction or truth")
    if not np.allclose(saved[FORECASTS], reproduced[FORECASTS], atol=1e-10, rtol=1e-12):
        raise ValueError("Original T7 expected errors differ from the saved weights' predictions")
    return saved.reset_index()


def _check_original_metrics(result, archived):
    expected = archived["methods"]["T7"]
    for key, value in expected.items():
        actual = result["original"][key]
        if value is None or actual is None:
            match = actual is value
        else:
            match = np.isclose(actual, value, atol=1e-9, rtol=1e-12)
        if not match:
            raise ValueError(f"Original T7 evaluation no longer reproduces {key}")


def run(workspace, study_root, official_dir, original_t7_dir, output_dir):
    from models.t7_error_regression.full_comparison import compare_decisions

    workspace, study, official_dir, original_dir, output = map(Path, (workspace, study_root, official_dir, original_t7_dir, output_dir))
    if output.exists():
        raise FileExistsError(f"Choose a new full T7 output directory: {output}")
    old = _json(T7_REPORT)
    matched, full, samples = load_samples(study)
    _, reserved, official, inputs = load_inputs(workspace, study, official_dir)
    archived_samples = _json(SAMPLE_SIZE_REPORT)["provenance"]
    if (samples["full"] != archived_samples["full"] or samples["added"] != archived_samples["added"]
            or samples["matched"]["training_sample_sha256"] != old["inputs"]["training_sample_sha256"]
            or len(matched) != old["training"]["n"] or inputs != old["inputs"]):
        raise ValueError("Training samples or evaluation sources differ from the archived T7 experiments")
    if REGRESSOR_PARAMS != old["training"]["regressor_parameters"] or REGRESSOR_PARAMS["max_iter"] != 80:
        raise ValueError("Full T7 must keep the original 80-stage regressor settings")
    rows = pd.concat([reserved.assign(split=reserved.region), official], ignore_index=True)
    if rows.id.duplicated().any() or full.id.isin(rows.id).any():
        raise ValueError("Training and evaluation must use disjoint photo IDs")
    baseline = _original_decisions(original_dir, old, rows)
    print(f"Verified original T7 on {len(rows):,} photos; fitting on {len(full):,} development photos", flush=True)
    full_model = fit(full[[*INPUT_COLUMNS, *ERRORS]])
    if full_model.t4.n_iter_ != 80 or full_model.t5.n_iter_ != 80:
        raise RuntimeError("Both full T7 regressors must complete 80 stages")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "results"
        stage.mkdir()
        weights = stage / "T7_full.joblib"
        joblib.dump(full_model, weights, compress=3)
        decisions = _decisions(joblib.load(weights), rows)
        comparison = compare_decisions(baseline, decisions)
        _check_original_metrics(comparison["new_final"], old["new_final"])
        _check_original_metrics(comparison["official"]["pooled"], old["official"]["pooled"])
        for key in TEST_SPLITS:
            _check_original_metrics(comparison["official"]["splits"][key], old["official"]["splits"][key])
        old_columns = [*FORECASTS, "T7_choose_T4", *SELECTED]
        shared = baseline.set_index("id")[old_columns].add_prefix("original_")
        decisions.join(shared, on="id", validate="one_to_one").to_parquet(stage / "decisions.parquet", index=False)
        summary = {
            "protocol": "Refit the original two 80-stage error regressors on all seven development regions. "
                        "Use the original held-out photos and frozen T4/T5 predictions. Each official photo counts once "
                        "in the pooled result. The reserved development region is reported separately.",
            "training": {"n": len(full), "regressor_parameters": REGRESSOR_PARAMS, "input_columns": list(INPUT_COLUMNS)},
            "inputs": {"training_samples": samples, "evaluation": inputs, "original_artifacts_sha256": old["output_sha256"],
                       "original_report_sha256": sha256(T7_REPORT), "sample_size_report_sha256": sha256(SAMPLE_SIZE_REPORT)},
            "output_sha256": {"T7_full.joblib": sha256(weights), "decisions.parquet": sha256(stage / "decisions.parquet")},
            **comparison,
        }
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    return summary


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "study-root", "official-dir", "original-t7-dir", "output-dir"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    run(**vars(args))
    print("COMPLETE", args.output_dir / "summary.json", flush=True)
