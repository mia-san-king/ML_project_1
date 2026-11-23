"""Train T7 on the archived seven-region sample and compare against frozen T6."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from models.clip_regression.scripts.compare_t1 import ERROR_ATOL_KM
from models.t6_router.coverage_gate import ERRORS, SEVEN, USED, _signature
from models.t6_router.evaluate import TEST_SPLITS
from models.t6_router.evaluate_coverage_official import _official_inputs
from models.t6_router.expanded_regions import FINAL, SEVEN as DEV_REGIONS, matched_samples
from models.t6_router.geographic_gate import POSITION
from models.t6_router.run_coverage_study import read_fold
from scripts.confidence.evaluate import sha256


ARCHIVED_REPORTS = Path(__file__).resolve().parents[2] / "report/experiments/2026-11-22-local-models/t6"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _with_t6(frame, path, digest, *, cohort):
    if sha256(path) != digest:
        raise ValueError("T6 per-photo decisions differ from the archived digest")
    source = frame.set_index("id", verify_integrity=True)
    saved = pd.read_parquet(path).set_index("id", verify_integrity=True)
    if not saved.index.sort_values().equals(source.index.sort_values()):
        raise ValueError("T6 evaluation IDs differ")
    saved = saved.loc[source.index]
    if not saved[cohort].eq(source[cohort]).all() or not saved.block.eq(source.block).all():
        raise ValueError("T6 evaluation split, region or block differs")
    if not np.array_equal(saved[list(POSITION)], source[list(POSITION)]):
        raise ValueError("T6 expert coordinates differ")
    if not np.allclose(saved[ERRORS], source[ERRORS], rtol=0, atol=ERROR_ATOL_KM):
        raise ValueError("T6 expert errors differ")
    p = saved[f"{SEVEN}_p_T4"].to_numpy(dtype=float)
    picks = saved[f"{SEVEN}_choose_T4"].to_numpy()
    if (picks.dtype.kind != "b" or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
            or not np.array_equal(picks, p > .5)):
        raise ValueError("T6 choices differ from its frozen closer scores")
    error = np.where(picks, saved.T4_error_km, saved.T5_error_km)
    if not np.array_equal(error, saved[f"{SEVEN}_error_km"]):
        raise ValueError("T6 selected errors differ from the recorded expert")
    if cohort == "split":
        for key in ("lat_pred", "lon_pred"):
            if not np.array_equal(np.where(picks, saved[f"T4_{key}"], saved[f"T5_{key}"]), saved[f"{SEVEN}_{key}"]):
                raise ValueError("T6 selected coordinates differ from the recorded expert")
    return source.assign(T6_choose_T4=picks).reset_index()


def load_inputs(workspace, study_root, official_dir):
    gate = study_root / "gate"
    study = _json(ARCHIVED_REPORTS / "matched_region_coverage.json")
    published = _json(ARCHIVED_REPORTS / "coverage_official.json")
    if _json(study_root / "study.json") != study or _json(official_dir / "summary.json") != published:
        raise ValueError("Run summaries differ from the archived T6 experiment")
    result = study["comparison"]
    if _json(gate / "summary.json") != result or published["selected_model"] != SEVEN:
        raise ValueError("Expected the archived seven-region T6 selector")
    by_region = {}
    for name in (*DEV_REGIONS, FINAL):
        frame, receipt = read_fold(study_root, name, study_root / "splits" / f"{name}.csv.gz")
        if receipt != study["folds"][name]:
            raise ValueError(f"{name}: expert receipt differs from the archived study")
        by_region[name] = frame
    held = by_region.pop(FINAL)
    train = matched_samples(pd.concat(by_region.values(), ignore_index=True),
                            budget=result["development_n_per_arm"], seed=study["sampling_seed"])["seven"]
    training_digest = _signature(train, USED)
    if training_digest != result["training"][SEVEN]["sample_sha256"] or _signature(held, USED) != result["final_input_sha256"]:
        raise ValueError("Seven-region training or reserved rows differ from the archived T6 inputs")
    held = _with_t6(held, gate / "decisions.parquet", result["outputs_sha256"]["decisions.parquet"], cohort="region")

    paired, raw, source_hashes = _official_inputs(workspace)
    model_file = f"{SEVEN}.joblib"
    gate_digest = sha256(gate / model_file)
    hashes = {**source_hashes, model_file: gate_digest, "coverage_summary": sha256(gate / "summary.json")}
    if (gate_digest != result["outputs_sha256"][model_file]
            or any(value != published["input_sha256"].get(name) for name, value in hashes.items())):
        raise ValueError("Official data or T6 comparator differs from its archived digest")
    official = _with_t6(paired.join(raw, validate="one_to_one").reset_index(), official_dir / "decisions.parquet",
                        published["decisions_sha256"], cohort="split")
    if set(train.id) & set(held.id) or set(train.id) & set(official.id) or set(held.id) & set(official.id):
        raise ValueError("Training, reserved region and official test photos must be disjoint")
    provenance = {"training_sample_sha256": training_digest, "new_final_sample_sha256": result["final_input_sha256"],
                  "new_final_decisions_sha256": result["outputs_sha256"]["decisions.parquet"],
                  "official_decisions_sha256": published["decisions_sha256"], "t6_input_sha256": hashes,
                  "regional_study_sha256": sha256(study_root / "study.json"),
                  "official_summary_sha256": sha256(official_dir / "summary.json")}
    return train, held, official, provenance


def run(workspace, study_root, official_dir, output_dir):
    from models.t7_error_regression.model import INPUT_COLUMNS, REGRESSOR_PARAMS, fit, predict
    from models.t7_error_regression.metrics import evaluate

    workspace, study, official_dir, output = map(Path, (workspace, study_root, official_dir, output_dir))
    if output.exists():
        raise FileExistsError(f"Choose a new T7 output directory: {output}")
    train, reserved, official, provenance = load_inputs(workspace, study, official_dir)
    frames = []
    with threadpool_limits(limits=4):
        model = fit(train[[*INPUT_COLUMNS, *ERRORS]])
        for rows, splits in ((reserved, reserved.region), (official, official.split)):
            expected = predict(model, rows.loc[:, list(INPUT_COLUMNS)])
            if not expected.index.equals(rows.index):
                raise ValueError("T7 predictions are misaligned with the input photos")
            columns = ["id", "block", *POSITION, *ERRORS, "T6_choose_T4"]
            frame = rows[columns].assign(split=splits).join(expected, validate="one_to_one")
            for key in ("lat_pred", "lon_pred", "error_km"):
                frame[f"T7_{key}"] = np.where(frame.T7_choose_T4, frame[f"T4_{key}"], frame[f"T5_{key}"])
            frame["T6_error_km"] = np.where(frame.T6_choose_T4, frame.T4_error_km, frame.T5_error_km)
            frames.append(frame)
    held, formal = frames
    summary = {"training": {"n": len(train), "photos_by_region": train.region.value_counts().sort_index().to_dict(),
                            "input_columns": list(INPUT_COLUMNS), "regressor_parameters": REGRESSOR_PARAMS},
               "protocol": "Two error regressors fit on the original seven-region T6 development sample. "
                           "The reserved region and official tests are evaluated separately against their archived T6 choices.",
               "new_final": evaluate(held),
               "official": {"splits": {name: evaluate(formal.loc[formal.split.eq(name)]) for name in TEST_SPLITS},
                            "pooled": evaluate(formal)}, "inputs": provenance}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "results"
        stage.mkdir()
        joblib.dump(model, stage / "T7.joblib", compress=3)
        pd.concat(frames, ignore_index=True).to_parquet(stage / "decisions.parquet", index=False)
        summary["output_sha256"] = {name: sha256(stage / name) for name in ("T7.joblib", "decisions.parquet")}
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    return summary


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for option in ("workspace", "study-root", "official-dir", "output-dir"):
        cli.add_argument("--" + option, type=Path, required=True)
    args = cli.parse_args()
    run(**vars(args))
    print("COMPLETE", args.output_dir / "summary.json", flush=True)
