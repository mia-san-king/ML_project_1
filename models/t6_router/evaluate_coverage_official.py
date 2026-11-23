"""Evaluate the frozen three- and seven-region T6 selectors on the three official tests."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from models.t6_router.coverage_gate import SEVEN, THREE
from models.t6_router.error_distribution_data import load_test_comparison
from models.t6_router.evaluate import TEST_SPLITS, distance_metrics as location_metrics
from models.t6_router.geographic_gate import NONPOSITION, POSITION
from models.t6_router.region_gate import confidence_scores, paired_change
from models.t6_router.regional_closer import distance_metrics, paired_distances
from models.t6_router.validate_spatial import ARCHIVE, CONFIDENCE
from scripts.confidence.evaluate import FEATURES, sha256


UPPER_EDGES = list(range(25, 251, 25))
EXPERTS = ("T1", "T3", "T4", "T5")


def _official_inputs(workspace):
    inputs_key = f"{CONFIDENCE}/inputs.parquet"
    predictions_key = f"{CONFIDENCE}/predictions.parquet"
    archive = pd.read_csv(ARCHIVE, sep="\t").query("root == 'workspace'").set_index("path").sha256
    raw_hash = sha256(workspace / inputs_key)
    if raw_hash != archive.at[inputs_key]:
        raise ValueError("Official raw confidence input digest mismatch")
    paired = load_test_comparison(workspace / predictions_key, workspace=workspace).set_index("id", verify_integrity=True)
    raw = pd.DataFrame(index=paired.index)
    for model in ("T4", "T5"):
        columns = FEATURES[model]
        rows = pd.read_parquet(workspace / inputs_key, columns=["id", "split", "block", *columns],
                               filters=[("model", "=", model), ("split", "in", TEST_SPLITS)])
        rows = rows.set_index("id", verify_integrity=True).sort_index()
        if (not rows.index.equals(paired.index) or not rows.split.eq(paired.split).all()
                or not rows.block.eq(paired.block).all()):
            raise ValueError(f"Official {model} raw feature IDs, splits or blocks do not align with predictions")
        raw[columns] = rows[columns]
    # load_test_comparison has verified the archived hashes of the prediction file and all three references.
    keys = (predictions_key, inputs_key, "data/t1_original/meta/smapshot.parquet",
            "data/t1_original/features/clip_vitl14_ids.npy", "data/splits/main.csv.gz")
    return paired, raw, {key: archive.at[key] for key in keys}


def _apply(paired, raw, model, arm):
    probabilities = confidence_scores(raw, model["confidence"])
    inputs = paired.loc[:, list(POSITION)].copy()
    inputs[list(NONPOSITION)] = probabilities
    p4 = np.asarray(model["gate"].predict_t4_probability(inputs), dtype=float)
    if p4.shape != (len(paired),) or not np.isfinite(p4).all() or ((p4 < 0) | (p4 > 1)).any():
        raise ValueError(f"{arm}: invalid closer-expert probabilities")
    picks = p4 > .5
    decisions = pd.DataFrame(index=paired.index)
    decisions[[f"{arm}_{key}" for key in NONPOSITION]] = probabilities
    decisions[f"{arm}_p_T4"], decisions[f"{arm}_choose_T4"] = p4, picks
    for key in ("lat_pred", "lon_pred", "error_km"):
        decisions[f"{arm}_{key}"] = np.where(picks, paired[f"T4_{key}"], paired[f"T5_{key}"])
    return decisions


def _results(frame):
    methods = {}
    for name in (*EXPERTS, THREE, SEVEN):
        errors = frame[f"{name}_error_km"].to_numpy()
        if name in (THREE, SEVEN):
            summary = distance_metrics(frame, frame[f"{name}_choose_T4"].to_numpy())
        elif name in ("T4", "T5"):
            summary = distance_metrics(frame, np.full(len(frame), name == "T4", dtype=bool))
        else:
            summary = location_metrics(errors)
        bins = np.searchsorted(UPPER_EDGES, errors, side="left")
        summary["histogram_counts"] = np.bincount(bins, minlength=len(UPPER_EDGES) + 1).tolist()
        methods[name] = summary
    new, old = (frame[f"{name}_choose_T4"].to_numpy() for name in (SEVEN, THREE))
    changes = paired_distances(frame, new, old)
    changes["hit25_changes"] = paired_change(frame, new, old)
    return {"n": len(frame), "n_blocks": int(frame.block.nunique()), "methods": methods, "seven_vs_three": changes}


def run(workspace, gate_dir, output_dir):
    workspace, gate_dir, output = map(Path, (workspace, gate_dir, output_dir))
    if output.exists():
        raise FileExistsError(f"Choose a new official evaluation directory: {output}")
    receipt = gate_dir / "summary.json"
    expected = json.loads(receipt.read_text(encoding="utf-8"))["outputs_sha256"]
    models, hashes = {}, {"coverage_summary": sha256(receipt)}
    for arm in (THREE, SEVEN):
        path = gate_dir / f"{arm}.joblib"
        hashes[path.name] = sha256(path)
        if hashes[path.name] != expected[path.name]:
            raise ValueError(f"{arm}: gate and calibrator digest mismatch")
        models[arm] = joblib.load(path)
    paired, raw, data_hashes = _official_inputs(workspace)
    base = ["split", "block", "latitude", "longitude",
            *(f"{name}_{key}" for name in EXPERTS for key in ("lat_pred", "lon_pred", "error_km"))]
    with threadpool_limits(limits=4):
        decisions = pd.concat([paired[base], *(_apply(paired, raw, models[arm], arm) for arm in (THREE, SEVEN))], axis=1)
    report = {"selected_model": SEVEN, "reference_model": THREE,
              "protocol": "Preselected seven-region T6. Both frozen selectors use their own frozen confidence "
                          "calibrators and the same production +8192 expert coordinates. Choose T4 when P(T4 closer) > 0.5.",
              "histogram_upper_edges_km": UPPER_EDGES, "histogram_rule": "Upper-inclusive bins, final count above 250 km.",
              "input_sha256": {**hashes, **data_hashes},
              "splits": {split: _results(decisions.loc[decisions.split.eq(split)]) for split in TEST_SPLITS},
              "pooled": _results(decisions)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "evaluation"
        stage.mkdir()
        predictions = stage / "decisions.parquet"
        decisions.reset_index().to_parquet(predictions, index=False)
        report["decisions_sha256"] = sha256(predictions)
        (stage / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace", required=True, type=Path, help="Windows project workspace containing the official archives")
    cli.add_argument("--gate-dir", required=True, type=Path, help="Completed coverage study's gate directory")
    cli.add_argument("--output-dir", required=True, type=Path, help="New official evaluation directory")
    args = cli.parse_args()
    result = run(**vars(args))
    print(f"Evaluated {result['pooled']['n']:,} official test photos. Results in {args.output_dir}", flush=True)
