"""Train and evaluate a joint selector on the frozen T4/T5 validation outputs."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from models.t6_router.geographic_gate import evaluate_gate, fit_gate
from models.t6_router.validate_spatial import ROOT, load_expert_validation
from scripts.confidence.evaluate import sha256


PRIOR_REPORT = ROOT / "report/experiments/2026-11-22-local-models/t6/spatial_validation.json"


def train(workspace, previous_decisions, output):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    previous_hash = sha256(previous_decisions)
    prior = json.loads(PRIOR_REPORT.read_text(encoding="utf-8"))
    if previous_hash != prior["decisions_sha256"]:
        raise ValueError("Expected the archived validation folds and out-of-fold confidence values")
    frame, _, _, hashes = load_expert_validation(workspace)
    frozen = pd.read_parquet(previous_decisions)
    with threadpool_limits(limits=4):
        report, decisions = evaluate_gate(frame, frozen)
        # Each row for the final fit uses confidence from the previous five-fold
        # study, generated while withholding that photo's entire geographic block.
        aligned = frozen.set_index("id", verify_integrity=True).loc[frame.id]
        fit_rows = frame.assign(T4_p25=aligned.T4_refit_p25.to_numpy(),
                                T5_p25=aligned.T5_refit_p25.to_numpy())
        gate = fit_gate(fit_rows)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as temp:
            stage = Path(temp) / "result"
            stage.mkdir()
            artifact = stage / "gate.joblib"
            joblib.dump(gate, artifact, compress=3)
            probe = fit_rows.iloc[::73]
            np.testing.assert_array_equal(joblib.load(artifact).predict_t4_probability(probe),
                                          gate.predict_t4_probability(probe))
            decisions.to_parquet(stage / "decisions.parquet", index=False)
            report.update({
                "input_sha256": hashes, "previous_decisions_sha256": previous_hash,
                "sklearn_version": sklearn.__version__, "gate_sha256": sha256(artifact),
                "decisions_sha256": sha256(stage / "decisions.parquet"),
                "final_fit": "All validation photos with exactly one hit, using out-of-fold branch confidences; deploy with the frozen T4/T5 confidence models.",
                "scope": "Development validation only. T5 previously used this validation set to select its reranking weight.",
            })
            (stage / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            stage.rename(output)
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace", type=Path, default=ROOT)
    cli.add_argument("--previous-decisions", required=True, type=Path)
    cli.add_argument("--output-dir", required=True, type=Path)
    args = cli.parse_args()
    outcome = train(args.workspace, args.previous_decisions, args.output_dir)
    print(json.dumps(outcome["methods"], indent=2))
